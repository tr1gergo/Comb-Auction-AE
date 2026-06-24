import io
p = r"c:\Users\User\Documents\kutatas\Simulations\comb-auctions\AIRPORT_simulations.ipynb"
with io.open(p, 'r', encoding='utf-8') as f:
    s = f.read()
replacements = [
    ("airport_manipulation_item_stats_all_configs_NEW.csv", "airport_data/airport_manipulation_item_stats_all_configs_NEW.csv"),
    ("airport_manipulation_summary_results_NEW.npy", "airport_data/airport_manipulation_summary_results_NEW.npy"),
    ("airport_manipulation_pct_histories_NEW.npy", "airport_data/airport_manipulation_pct_histories_NEW.npy"),
    ("airport_manipulation_success_histories_NEW.npy", "airport_data/airport_manipulation_success_histories_NEW.npy"),
    ("airport_manipulation_item_stats_all_configs.csv", "airport_data/airport_manipulation_item_stats_all_configs.csv"),
    ("airport_manipulation_summary_results.npy", "airport_data/airport_manipulation_summary_results.npy"),
    ("airport_manipulation_pct_histories.npy", "airport_data/airport_manipulation_pct_histories.npy"),
    ("airport_manipulation_success_histories.npy", "airport_data/airport_manipulation_success_histories.npy"),
    ("airport_simulation_results.csv", "airport_data/airport_simulation_results.csv"),
]
for a,b in replacements:
    s = s.replace(a,b)
with io.open(p, 'w', encoding='utf-8') as f:
    f.write(s)
print('patched')
