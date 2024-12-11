# echo "1/18 Running santos inference..."
# python extractVectors.py \
# --benchmark santos \
# --table_order column \
# --run_id 0 \
# --return_serialized \
# --save_model

# echo "3/18 Running santos-p-col inference..."
# python extractVectors.py \
# --benchmark santos-p-col \
# --table_order column \
# --run_id 0 \
# --return_serialized \
# --save_model


echo "5/18 Running tus inference..."
python extractVectors.py \
--benchmark tus \
--table_order column \
--run_id 0 \
--return_serialized \
--save_model


echo "7/18 Running tus-p-col inference..."
python extractVectors.py \
--benchmark tus-p-col \
--table_order column \
--run_id 0 \
--return_serialized \
--save_model



echo "9/18 Running tusLarge inference..."
python extractVectors.py \
--benchmark tusLarge \
--table_order column \
--run_id 0 \
--return_serialized \
--save_model

echo "11/18 Running tusLarge-p-col inference..."
python extractVectors.py \
--benchmark tusLarge-p-col \
--table_order column \
--run_id 0 \
--return_serialized \
--save_model




echo "13/18 Running pylon inference..."
python extractVectors.py \
--benchmark pylon \
--table_order column \
--run_id 0 \
--return_serialized \
--save_model


echo "15/18 Running pylon-p-col inference..."
python extractVectors.py \
--benchmark pylon-p-col \
--table_order column \
--run_id 0 \
--return_serialized \
--save_model




# # # echo "17/18 Running ugen_v1 inference..."
# # # python extractVectors.py \
# # # --benchmark ugen_v1 \
# # # --table_order column \
# # # --run_id 0 \
# # # --save_model

# # # echo "18/18 Running ugen_v2 inference..."
# # # python extractVectors.py \
# # # --benchmark ugen_v2 \
# # # --table_order column \
# # # --run_id 0 \
# # # --save_model

# # echo "All inference tasks completed!"