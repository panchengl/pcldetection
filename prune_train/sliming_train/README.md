In this section, the code use sliming to prune network

    first: python main_retinanet_prune_sliming.py

            use l1 norm make weights  sparsed. when u get the stable map, u can use nest stage

    second: python main_retinanet_prune_finetune.py

            use this code finetune sparsed net.

            becarful: if u want use knowledge distill to finetune, u can use: python main_retinanet_prune_finetune_knowledge_distill_loos2.py

