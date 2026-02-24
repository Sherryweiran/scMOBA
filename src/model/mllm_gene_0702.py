import torch
from torch import nn
import math
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoModel, AutoModelForCausalLM
from transformers import BertModel, BertTokenizer
import sys
root =  "/fs-computility/mabasic/weiran/scLaGene/"
sys.path.append(root)
from src.model.muti_cross_attention import CrossMultiHeadAttention
from src.model.GCN import aff_to_adj_batch, GraphConvolution, GCN
import pickle

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class scLaGene(nn.Module):
    def __init__(self, args):
        super(scLaGene, self).__init__()
        self.args = args

        self.bert_encoder = BertModel.from_pretrained(args.bert_path)

        # 根据需要冻结BERT模型的参数
        if args.bert_frozen:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False

        self.GCN_module = GCN(args.bert_hidden_size, args.GCN_embedding_dim, args.llm_hidden_size, args.dropout)

        self.learned_query = nn.Parameter(torch.rand(args.num_querys, self.args.llm_hidden_size), requires_grad=True)
        self.abstractor = CrossMultiHeadAttention(self.args.llm_hidden_size, args.encoder_num_heads, att_dropout=0.0)

        self.llm = AutoModelForCausalLM.from_pretrained(args.llm_path)
        self.llm.resize_token_embeddings(args.vocab_size)
        self.llm.enable_input_require_grads()
        self.llm.requires_grad_(not args.llm_frozen)
        self.llm.model.embed_tokens.requires_grad_(True)
    def forward(self, images, input_ids, labels, attention_masks, original_composition):

        # Original processing for single sequence
        #print("images min/max:", images.min(), images.max())
        #print("BERT vocab_size:", self.bert_encoder.config.vocab_size)
    
         # 检查是否存在负值或超出词汇表的值
        assert (images >= 0).all(), "images 包含负值！"
        assert (images < self.bert_encoder.config.vocab_size).all(), "images 超出词汇表范围！"
        try:
            bert_output = self.bert_encoder(input_ids=images)
        except Exception as e:  # 捕获所有异常，不限定类型
           
            print(f"Original error: {str(e)}")
            print(f"Original images tensor info:")
            print(f"  shape: {images.shape}")
            print(f"  dtype: {images.dtype}")
            print(f"  sample values: {images[0][:5] if len(images.shape) > 1 else images[:5]}\n")

            # 强制转换并重试
            images = images.to(torch.int64)
            print(f"Converted images tensor info:")
            print(f"  dtype: {images.dtype}")
            print(f"  sample values: {images[0][:5] if len(images.shape) > 1 else images[:5]}\n")

            #bert_output = self.bert_encoder(input_ids=images)  # 重试

        bert_hidden_state = bert_output.last_hidden_state 

        adj = aff_to_adj_batch(bert_hidden_state)
        GCN_results = self.GCN_module(bert_hidden_state, adj)

        batch_size = GCN_results.size(0)
        learned_query = self.learned_query.unsqueeze(0).expand(batch_size, -1, -1)
        image_feats = self.abstractor(learned_query, GCN_results)
        #print("org:",original_composition)
        #print("img_shape:",image_feats.shape)

        # world_size = dist.get_world_size()
        # # 为 all_gather 准备 buffer
        # feats_gather = [torch.zeros_like(image_feats) for _ in range(world_size)]
        # # 把各卡的 image_feats 都收集到每个进程的 feats_gather
        # dist.all_gather(feats_gather, image_feats)
        # # 拼回全局
        # global_image_feats = torch.cat(feats_gather, dim=0)  # [global_N, Q, D]

        groups = torch.split(image_feats, original_composition, dim=0)
 
        aggregated_groups = []
        for group, num in zip(groups, original_composition):
            # Step 2: 对标记为10的组取平均，否则直接保留
            if num == 10:
                group = group.mean(dim=0, keepdim=True)  # [1, 256, 2048]
            aggregated_groups.append(group)
    
        # Step 3: 拼接所有组
        image_feats = torch.cat(aggregated_groups, dim=0) 

        # Rest of the processing remains the same
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat(
            (image_feats, inputs_embeds[:, image_feats.shape[1]:, :]), dim=1)

        output = self.llm(inputs_embeds=inputs_embeds, 
                        attention_mask=attention_masks, 
                        labels=labels)

        return {
            "loss": output["loss"],
            "logits": output["logits"]
        }
    




    def generate(self, images, input_ids):
        with torch.no_grad():
            # 使用BERT的编码器处理输入文本
            bert_output = self.bert_encoder(input_ids=images)
            bert_hidden_state = bert_output.last_hidden_state  # 获取BERT的最后一层隐藏状态

            # 将BERT的输出传递给GCN
            adj = aff_to_adj_batch(bert_hidden_state)
            GCN_results = self.GCN_module(bert_hidden_state, adj)

            batch_size = GCN_results.size(0)
            learned_query = self.learned_query.unsqueeze(0).expand(batch_size, -1, -1)

            # 使用抽象器处理GCN输出
            image_feats = self.abstractor(learned_query, GCN_results)

            # 将image_feats插入到LLM的输入嵌入中
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat(
                (image_feats, inputs_embeds[:, image_feats.shape[1]:, :]), dim=1)  # 将image_feats插入到文本嵌入中

            # 使用LLM生成输出
            output_ids = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=256, top_k=50)
            return output_ids
        
    def get_image_feats(self, images, input_ids):

        #提取图像特征
      
        with torch.no_grad():
            bert_output = self.bert_encoder(input_ids=images)
            bert_hidden_state = bert_output.last_hidden_state  # 获取BERT的最后一层隐藏状态

            # 将BERT的输出传递给MLP Projector
            #image_feats = self.mlp_projector(bert_hidden_state)
            adj = aff_to_adj_batch(bert_hidden_state)
        
            # 将BERT的输出传递给GCN
            GCN_results=self.GCN_module(bert_hidden_state,adj)
            batch_size = GCN_results.size(0)  
            learned_query = self.learned_query.unsqueeze(0).expand(batch_size, -1, -1)

         
            #image_feats = self.mlp_projector(bert_hidden_state)
            #GCN output  to abstractor
            image_feats = self.abstractor(learned_query,GCN_results)

        return image_feats
    

    def compute_attention_scores_with_node_info(self, images, input_ids,attention_masks):
        """
        计算以下内容：
         1. LLM 第一层中最后一个 token 和 learned_query 的注意力分数。
        2. learned_query 和 GCN 节点的注意力分数。
        3. 对于每个 node 计算从不同 query 来的注意力分数的加权和。
        4. 对于每个 query 计算来自最后一个 token 的注意力分数。
        5. 记录每个 node 的位置信息，并根据位置检索对应的 input_id。
        6. 计算 node_final_scores 在 batch_size 维度上的平均值。

        参数:
        - self: scLaGene 模型实例
        - images: 输入图像
        - input_ids: 输入文本的 token IDs

        返回:
        - last_token_query_attention: 最后一个 token 和每个 query 的注意力分数 [batch_size, num_queries]
        - query_node_attention: 每个 query 和每个 node 的注意力分数 [batch_size, num_queries, num_nodes]
        - node_final_scores: 每个 node 的最终加权注意力分数 [batch_size, num_nodes]
         node_final_scores_mean: node_final_scores 在 batch_size 维度上的平均值 [num_nodes]
        - node_positions: 每个 node 的位置信息 [batch_size, num_nodes]
        - node_input_ids: 每个 node 对应的 input_id [batch_size, num_nodes]
        - query_final_scores: 每个 query 的来自最后一个 token 的注意力分数 [batch_size, num_queries]
        """
        with torch.no_grad():
          # 1. 获取 BERT 的输出
            bert_output = self.bert_encoder(input_ids=images)
            bert_hidden_state = bert_output.last_hidden_state

            # 2. 获取 GCN 的输出
            adj = aff_to_adj_batch(bert_hidden_state)
            GCN_results = self.GCN_module(bert_hidden_state, adj)

            # 3. 获取 image_feats
            batch_size = GCN_results.size(0)
            learned_query = self.learned_query.unsqueeze(0).expand(batch_size, -1, -1)
            image_feats = self.abstractor(learned_query, GCN_results)

            # 4. 获取 inputs_embeds
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat((image_feats, inputs_embeds[:, image_feats.shape[1]:, :]), dim=1)

            # 5. 获取 LLM 的第一层的注意力分数
            #llm_layer1 = self.llm.model.model.layers[0] 
            #output = llm_layer1(inputs_embeds)
            #attention_scores = output.attentions  # 假设 LLM 的第一层返回了注意力分数 [batch_size, num_heads, seq_len, seq_len]
            #output = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_masks, labels=labels)
            #output_ids = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=256, top_k=50)
            outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_masks, output_attentions=True)
            attentions = outputs.attentions
            layer_0_attention = attentions[0]
            # 6. 提取最后一个 token 和 learned_query 的注意力分数
            # 假设 learned_query 对应 inputs_embeds 的前 num_queries 个 token
            num_queries = learned_query.size(1)
            last_token_attention = layer_0_attention[:, :, -1, :num_queries]  # [batch_size, num_heads, num_queries]
            last_token_attention = last_token_attention.mean(dim=1)  # 对多头取均值 [batch_size, num_queries]

            # 7. 获取 abstractor 的交叉注意力分数（learned_query 和 GCN 节点）
            def abstractor_with_attention(abstractor, query, context):
                Q = abstractor.Wq(query)
                K = abstractor.Wk(context)
                V = abstractor.Wv(context)

                batch_size = query.size(0)

                Q = Q.view(batch_size, -1, abstractor.num_heads, abstractor.depth).transpose(1, 2)
                K = K.view(batch_size, -1, abstractor.num_heads, abstractor.depth).transpose(1, 2)
                V = V.view(batch_size, -1, abstractor.num_heads, abstractor.depth).transpose(1, 2)

                att_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(abstractor.depth)
                att_weights = F.softmax(att_weights, dim=-1)

                output = torch.matmul(att_weights, V)
                output = output.transpose(1, 2).contiguous().view(batch_size, -1, abstractor.emb_dim)
                output = abstractor.fc(output)

                return output, att_weights
            
            def get_top_k_nodes_and_images(node_final_scores_mean, images, k):
    
                # 1. 找到最高的 k 个分数及其索引
                top_k_scores, top_k_indices = torch.topk(node_final_scores_mean, k=k)  # [k]

                # 2. 找到对应的 images 的值
                top_k_images=images[:, top_k_indices]  # [k]
         

                return  top_k_images

         # 获取 abstractor 的输出和注意力分数
            _, query_node_attention = abstractor_with_attention(self.abstractor, learned_query, GCN_results)
            query_node_attention = query_node_attention.mean(dim=1)  # 对多头取均值 [batch_size, num_queries, num_nodes]

            # 8. 计算每个 node 的最终加权注意力分数
            # 对于每个 node，加权和 = sum(last_token_query_attention * query_node_attention)
            node_final_scores = torch.einsum('bq,bqn->bn', last_token_attention, query_node_attention)  # [batch_size, num_nodes]

            # 9. 计算 node_final_scores 在 batch_size 维度上的平均值
            node_final_scores_mean = node_final_scores.mean(dim=0)  # [num_nodes]

        
            # 10. 记录每个 node 的位置信息
            # 假设 GCN_results 的每个 node 对应输入序列中的一个位置
            # 这里假设 GCN_results 的 node 顺序与输入序列的顺序一致
            num_nodes = GCN_results.size(1)
            #node_positions = torch.arange(num_nodes).unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_nodes]

            # 11. 根据 node 的位置检索对应的 input_id
            #node_input_ids = input_ids[:, :num_nodes]  # [batch_size, num_nodes]

            # 12. 计算每个 query 的来自最后一个 token 的注意力分数
            query_final_scores = last_token_attention  # [batch_size, num_queries]
            top_k_images = get_top_k_nodes_and_images(node_final_scores_mean, images, k=20)


            path="/lustre/usr/taolab/weir/miniconda3/envs/goodbai/lib/python3.12/site-packages/geneformer/token_dictionary.pkl"
            # load token dictionary (Ensembl IDs:token)
            with open(path, "rb") as f:
                 gene_token_dict = pickle.load(f)

            # 反转 gene_token_dic 变成 {索引: 基因名}
            index_to_gene = {v: k for k, v in gene_token_dict.items()}
            top_k_indices = top_k_images.cpu().tolist()[0]  # 转换到 CPU 并提取索引
            top_genes = [index_to_gene[idx] for idx in top_k_indices if idx in index_to_gene]

            return node_final_scores_mean, top_k_images,top_genes
            