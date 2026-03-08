import torch
from torch import nn
import math
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoModel, AutoModelForCausalLM
from transformers import BertModel, BertTokenizer
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

        assert (images >= 0).all(), "Gene data contains negative values."
        assert (images < self.bert_encoder.config.vocab_size).all(), "Gene token indices exceed the vocabulary size."
        try:
            bert_output = self.bert_encoder(input_ids=images)
        except Exception as e:  
           
            print(f"Original error: {str(e)}")
            print(f"Original gene tensor info:")
            print(f"  shape: {images.shape}")
            print(f"  dtype: {images.dtype}")
            print(f"  sample values: {images[0][:5] if len(images.shape) > 1 else images[:5]}\n")

            images = images.to(torch.int64)
            print(f"Converted gene tensor info:")
            print(f"  dtype: {images.dtype}")
            print(f"  sample values: {images[0][:5] if len(images.shape) > 1 else images[:5]}\n")


        bert_hidden_state = bert_output.last_hidden_state 

        adj = aff_to_adj_batch(bert_hidden_state)
        GCN_results = self.GCN_module(bert_hidden_state, adj)

        batch_size = GCN_results.size(0)
        learned_query = self.learned_query.unsqueeze(0).expand(batch_size, -1, -1)
        image_feats = self.abstractor(learned_query, GCN_results)

        groups = torch.split(image_feats, original_composition, dim=0)
 
        aggregated_groups = []
        for group, num in zip(groups, original_composition):
            # process st data neighbor 
            if num == 10:
                group = group.mean(dim=0, keepdim=True)  
            aggregated_groups.append(group)
    
        # Concatenate all groups
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
            bert_output = self.bert_encoder(input_ids=images)
            bert_hidden_state = bert_output.last_hidden_state  

            adj = aff_to_adj_batch(bert_hidden_state)
            GCN_results = self.GCN_module(bert_hidden_state, adj)

            batch_size = GCN_results.size(0)
            learned_query = self.learned_query.unsqueeze(0).expand(batch_size, -1, -1)

            image_feats = self.abstractor(learned_query, GCN_results)

            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat(
                (image_feats, inputs_embeds[:, image_feats.shape[1]:, :]), dim=1)  
            output_ids = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=256, top_k=50)
            return output_ids
        
    def get_image_feats(self, images, input_ids):

      
        with torch.no_grad():
            bert_output = self.bert_encoder(input_ids=images)
            bert_hidden_state = bert_output.last_hidden_state  

            adj = aff_to_adj_batch(bert_hidden_state)
        
            GCN_results=self.GCN_module(bert_hidden_state,adj)
            batch_size = GCN_results.size(0)  
            learned_query = self.learned_query.unsqueeze(0).expand(batch_size, -1, -1)

            image_feats = self.abstractor(learned_query,GCN_results)

        return image_feats

    def compute_attention_scores_with_node_info(self, images, input_ids,attention_masks):
      
        with torch.no_grad():
      
            bert_output = self.bert_encoder(input_ids=images)
            bert_hidden_state = bert_output.last_hidden_state
            adj = aff_to_adj_batch(bert_hidden_state)
            GCN_results = self.GCN_module(bert_hidden_state, adj)

    
            batch_size = GCN_results.size(0)
            learned_query = self.learned_query.unsqueeze(0).expand(batch_size, -1, -1)
            image_feats = self.abstractor(learned_query, GCN_results)

            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat((image_feats, inputs_embeds[:, image_feats.shape[1]:, :]), dim=1)
            if hasattr(self.llm.config, "output_attentions"):
                self.llm.config.output_attentions = True
          
            outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_masks, output_attentions=True)
            attentions = outputs.attentions
            layer_0_attention = attentions[0]
   
            num_queries = learned_query.size(1)
            last_token_attention = layer_0_attention[:, :, -1, :num_queries]  
            last_token_attention = last_token_attention.mean(dim=1)  
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
                

                if not torch.is_tensor(node_final_scores_mean):
                    node_final_scores_mean = torch.as_tensor(node_final_scores_mean)

    
                scores = node_final_scores_mean.flatten()
                
                n = int(scores.numel())
                if n == 0:
                    return (scores, torch.empty(0, dtype=torch.long, device=scores.device))
                k_eff = min(k, n)
          
                top_k_scores, top_k_indices = torch.topk(node_final_scores_mean, k=k_eff)  # [k]

                top_k_images=images[:, top_k_indices]  # [k]
         

                return  top_k_images

        
            _, query_node_attention = abstractor_with_attention(self.abstractor, learned_query, GCN_results)
            query_node_attention = query_node_attention.mean(dim=1)  
            node_final_scores = torch.einsum('bq,bqn->bn', last_token_attention, query_node_attention)  

            node_final_scores_mean = node_final_scores.mean(dim=0)  # [num_nodes]

            num_nodes = GCN_results.size(1)
            
            query_final_scores = last_token_attention  # [batch_size, num_queries]
            top_k_images = get_top_k_nodes_and_images(node_final_scores_mean, images, k=500)

            
            path="src/model/token_dictionary_human_RNA.pkl"

            # load token dictionary (Ensembl IDs:token)
            with open(path, "rb") as f:
                 gene_token_dict = pickle.load(f)
            index_to_gene = {v: k for k, v in gene_token_dict.items()}
            top_k_indices = top_k_images.cpu().tolist()[0] 
            top_genes = [index_to_gene[idx] for idx in top_k_indices if idx in index_to_gene]

            return node_final_scores_mean, top_k_images,top_genes

            