import yfinance as yf
import pandas as pd

if __name__=="__main__":
    
    tickers=[]
    for i in range(65,91):
        tickers.append(chr(i))
    for i in range(65,91):
        for j in range(65,91):
            tickers.append(chr(i)+chr(j))
    for i in range(65,91):
        for j in range(65,91):
            for k in range(65,91):
                tickers.append(chr(i)+chr(j)+chr(k))
    for i in range(65,91):
        for j in range(65,91):
            for k in range(65,91):
                for l in range(65,91):
                    tickers.append(chr(i)+chr(j)+chr(k)+chr(l))
    for i in range(65,91):
        for j in range(65,91):
            for k in range(65,91):
                for l in range(65,91):
                    for m in range(65,91):
                        tickers.append(chr(i)+chr(j)+chr(k)+chr(l)+chr(m))
    ticker_list=[]
    i=0
    for t in tickers[:]:
        try:
            i=i+1
            if i%500==0:
                print("--------------------------------------------------------------",i)
                print("--------------------------------------------------------------")
                print("--------------------------------------------------------------")
                with open('tickers_list_updated.csv', 'a') as f:
                    pd.DataFrame(ticker_list).to_csv(f, header=False)
                    ticker_list=[]
            tick=yf.Ticker(t)
            if len(tick.get_actions())>0:
                ticker_list.append(t)
        except KeyboardInterrupt:
            continue
    
