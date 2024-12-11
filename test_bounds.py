import pickle

from bounds import verify, upper_bound_bm, lower_bound_bm


tables = pickle.load(open("starmie_datalake_columns_santos.pkl","rb"))
queries = pickle.load(open("starmie_query_columns_santos.pkl","rb"))

query = queries[4]
threshold = 0.6
for table in tables:
    lb = lower_bound_bm(table[1], query[1], threshold)
    ub = upper_bound_bm(table[1], query[1], threshold)
    true = verify(table[1], query[1], threshold)
    print("lower bound: ", lb,"upper bound: ", ub, "true value: ", true)