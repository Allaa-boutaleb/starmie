# echo "1/6 Training Santos"
# CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
#   --task santos \
#   --batch_size 64 \
#   --lr 5e-5 \
#   --lm roberta \
#   --n_epochs 10 \
#   --max_len 128 \
#   --projector 768 \
#   --save_model \
#   --table_order column \
#   --augment_op drop_col \
#   --sample_meth tfidf_entity \
#   --fp16 \
#   --run_id 0

echo "2/6 Training TUS"
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
  --task tus \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 10 \
  --max_len 128 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op drop_cell \
  --sample_meth alphaHead \
  --fp16 \
  --run_id 0

echo "3/6 Training TUS Large"
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
  --task tusLarge \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 10 \
  --max_len 128 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op drop_cell \
  --sample_meth tfidf_entity \
  --fp16 \
  --run_id 0

echo "4/6 Training Pylon"
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
  --task pylon \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 10 \
  --max_len 128 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op drop_col \
  --sample_meth tfidf_entity \
  --fp16 \
  --run_id 0


# # echo "5/6 Training UGEN v1"
# # CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
# #   --task ugen_v1 \
# #   --batch_size 64 \
# #   --lr 5e-5 \
# #   --lm roberta \
# #   --n_epochs 10 \
# #   --max_len 128 \
# #   --projector 768 \
# #   --save_model \
# #   --table_order column \
# #   --augment_op drop_col \
# #   --sample_meth tfidf_entity \
# #   --fp16 \
# #   --run_id 0

# # echo "6/6 Training UGEN v2"
# # CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
# #   --task ugen_v2 \
# #   --batch_size 64 \
# #   --lr 5e-5 \
# #   --lm roberta \
# #   --n_epochs 10 \
# #   --max_len 128 \
# #   --projector 768 \
# #   --save_model \
# #   --table_order column \
# #   --augment_op drop_col \
# #   --sample_meth tfidf_entity \
# #   --fp16 \
# #   --run_id 0
