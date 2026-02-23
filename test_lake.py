import lakeapi
lakeapi.use_sample_data(anonymous_access=True)
df = lakeapi.load_data(table="book", start="2022-10-01", end="2022-10-02", symbols=["BTC-USDT"], exchanges=["BINANCE"])
print(df.columns)
print(df.head(1))
