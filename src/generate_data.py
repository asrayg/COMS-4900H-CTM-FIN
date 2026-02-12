from data.generator import L2BookGenerator
import os

def prepare_data():
    gen = L2BookGenerator(k_levels=10)
    print("Generating training data...")
    df_train = gen.generate(n_steps=20000) # ~16 mins of data at 50ms
    
    print("Generating validation data...")
    df_val = gen.generate(n_steps=5000, seed=99)
    
    print("Generating test data with shocks...")
    df_test = gen.generate(n_steps=5000, seed=101)
    df_test = gen.inject_shock(df_test, shock_type='spread', start_idx=2000, duration=100)
    df_test = gen.inject_shock(df_test, shock_type='liquidity', start_idx=4000, duration=100)
    
    os.makedirs('data', exist_ok=True)
    df_train.to_csv('data/train.csv')
    df_val.to_csv('data/val.csv')
    df_test.to_csv('data/test.csv')
    print("Datasets saved to data/ directory.")

if __name__ == "__main__":
    prepare_data()
