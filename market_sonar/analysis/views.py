from django.shortcuts import render
from .models import stockdata 
from .models import feedback as fd
from home.models import Client
from django.shortcuts import redirect
from django.urls import reverse
from django.views.decorators.cache import cache_control

# Create your views here.
import numpy as np
from nsepy import get_history
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import plotly.offline as opy
import plotly.graph_objects as go

from newsapi import NewsApiClient
import pandas as pd
from django.shortcuts import redirect
from nltk.sentiment.vader import SentimentIntensityAnalyzer


import datetime as dt
import snscrape.modules.twitter as sntwitter
import re
import matplotlib.pyplot as plt
import io
import urllib,base64



#-----------------------------Prediction and Graph-------------------------------------------------------------------------------------

def News(req):
    if(req.session.is_empty()): 
        
        req.session.set_expiry(0)      
        return render(req,'home/login.html') 
        
    else:
        total_score=0
        AVG=0
        newsapi = NewsApiClient(api_key='4e0ef55dfc634f40961b7249ae0128cf')
        News=[]
        vade=SentimentIntensityAnalyzer()
        if req.method=='POST':
            symbol=f"{req.POST['symbol']}"
            articles = newsapi.get_everything(q=symbol,
                                      language='en',
                                      sort_by='relevancy')
            
        else:          
            articles = newsapi.get_top_headlines(                                         
                                                category='business',
                                                language='en',
                                                country='in')
            
            
        for x in articles['articles']:
            score=((vade.polarity_scores(x['title'])['compound']+1)/2)*100
            total_score=total_score+score
            News.append([x['publishedAt'][:10],x['title'],score])
        l=len(articles['articles'])
            #df=pd.DataFrame(News,columns=['date','title'])
        AVG=total_score/l
        print(l)
        context={'d':News,'av':AVG}
        return render(req,'analysis/news.html',context)
        
def stocklist():
    df=pd.read_csv("analysis/stockdata/equity.csv")
    nse_list=df["SYMBOL"].tolist()
    return(nse_list)



def stock_prediction(req):
    if(req.session.is_empty()):
        
        req.session.set_expiry(0) 
        return render(req,'login.html') 
    else:
        if req.method == 'POST':
            symbol=f"{req.POST['symbol']}"
        else:
            symbol='TCS'
            
        import datetime 
        
        stock=stockdata()
        import math
        start=datetime.date(2020,1,1)
        end=datetime.date.today()
        st=stockdata.objects.filter(symbol=symbol)
        loc=f"analysis/stockdata/{symbol}.csv"
        
        if st.count()==1:
            for i in st:
                if i.date==datetime.date.today():
                    pass
                else:
                    df=pd.read_csv(loc)
                    lastdate=datetime.datetime.strptime(df['Date'].max(),"%Y-%m-%d")
                    print(type(lastdate))
                    start=lastdate.date()+datetime.timedelta(days = 1)
                    data=get_history(symbol=symbol, start=start , end=end)
                    data.to_csv(loc,mode='a',index=True,header=False)
                    i.date=datetime.date.today()
                    i.save()
                    
        else:
            data=get_history(symbol=symbol, start=start , end=end)
            data.to_csv(loc)
            stock.symbol=symbol
            stock.date=datetime.date.today()
            stock.save()
            
               
        df=pd.read_csv(loc)
        df=df.set_index('Date')
        data=df.filter(["Close"])
        dataset=data.values
        training_data_len=math.ceil(len(dataset)*0.8)
        scaler=MinMaxScaler(feature_range=(0,1))
        scaled_data=scaler.fit_transform(dataset)
        train_data=scaled_data[0:training_data_len,:]
        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model = Sequential()
        model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences = False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        model.fit(x_train, y_train, batch_size = 1, epochs = 1)
        test_data = scaled_data[training_data_len - 60: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range (60, len(test_data)):
             x_test.append(test_data[i - 60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid.insert(1,'predictions',predictions)
        last_60_days = data[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = train.index, y = train['Close'],
                            mode='lines',
                            name='Close',
                            marker_color = '#1F77B4'))
        fig.add_trace(go.Scatter(x = valid.index, y = valid['Close'],
                            mode='lines',
                            name='Val',
                            marker_color = '#FF7F0E'))
        fig.add_trace(go.Scatter(x = valid.index, y = valid.predictions,
                            mode='lines',
                            name='Predictions',
                            marker_color = '#2CA02C'))

        fig.update_layout(
            title=symbol,
            titlefont_size = 28,
            hovermode = 'x',
            xaxis = dict(
                title='Date',
                titlefont_size=16,
                tickfont_size=14),
            
            height = 800,
            
            yaxis=dict(
                title='Close price in INR (₹)',
                titlefont_size=16,
                tickfont_size=14),
            legend=dict(
                y=0,
                x=1.0,
                bgcolor='rgba(255, 255, 255, 0)',
                bordercolor='rgba(255, 255, 255, 0)'))

        div=opy.plot(fig,auto_open=False,output_type='div')
        nse_list=stocklist()
        context={'list':nse_list,'graph':div,'price':pred_price}
        return render(req,'analysis/graph.html',context)

def socio(req):
    if(req.session.is_empty()):
        
        req.session.set_expiry(0) 
        return redirect('home:Login')
    else:
        if req.method == 'POST':
            symbol=f"{req.POST['symbol']}"
            print(symbol)
        else:
            symbol='INDIA'
            
    query = symbol
    noOfTweet = 1000
    noOfDays = 7
            
    #Creating list to append tweet data
    tweets_list = []
    now = dt.date.today()
    now = now.strftime('%Y-%m-%d')
    yesterday = dt.date.today() - dt.timedelta(days = int(noOfDays))
    yesterday = yesterday.strftime('%Y-%m-%d')
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query + ' lang:en since:' +  yesterday + ' until:' + now + ' -filter:links -filter:replies').get_items()):
        if i > int(noOfTweet):
            break
        tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.username])
    df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
    df["Text"] = df["Text"].apply(cleanTxt)
    positive = 0
    negative = 0
    neutral = 0
    #Creating empty lists
    tweet_list1 = []
    neutral_list = []
    negative_list = []
    positive_list = []

    #Iterating over the tweets in the dataframe
    for tweet in df['Text']:
        tweet_list1.append(tweet)
        analyzer = SentimentIntensityAnalyzer().polarity_scores(tweet)
        neg = analyzer['neg']
        neu = analyzer['neu']
        pos = analyzer['pos']
        comp = analyzer['compound']

        if neg > pos:
            negative_list.append(tweet) #appending the tweet that satisfies this condition
            negative += 1 #increasing the count by 1
        elif pos > neg:
            positive_list.append(tweet) #appending the tweet that satisfies this condition
            positive += 1 #increasing the count by 1
        elif pos == neg:
            neutral_list.append(tweet) #appending the tweet that satisfies this condition
            neutral += 1 #increasing the count by 1 

    positive = percentage(positive, len(df)) #percentage is the function defined above
    negative = percentage(negative, len(df))
    neutral = percentage(neutral, len(df))
    #Converting lists to pandas dataframe
    tweet_list1 = pd.DataFrame(tweet_list1)
    neutral_list = pd.DataFrame(neutral_list)
    negative_list = pd.DataFrame(negative_list)
    positive_list = pd.DataFrame(positive_list)
    #using len(length) function for counting
    labels = ['Positive ['+str(round(positive))+'%]' , 'Neutral ['+str(round(neutral))+'%]','Negative ['+str(round(negative))+'%]']
    sizes = [positive, neutral, negative]
    colors = ['yellowgreen', 'blue','red']
    patches, texts = plt.pie(sizes,colors=colors, startangle=90)
    plt.style.use('default')
    plt.legend(labels)
    plt.title("Sentiment Analysis Result for keyword= "+query+"" )
    plt.axis('equal')
    fig=plt.gcf()
    buf=io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    return render(req,'analysis/socio.html',{'data':uri})

                    
                    


def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
    text = re.sub('#', '', text)  # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text)  # Removing RT
    text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
    return text

def percentage(part,whole):
    return 100 * float(part)/float(whole)


def feedback(req):
    
    if(req.session.is_empty()): 
        
        req.session.set_expiry(0)      
        return redirect('home:Login') 
    else:
        if req.method=='POST':
            symbol=f"{req.POST['T1']}"
            f=fd()
            id1= req.session['id']
            cred=Client.objects.filter(id=id1)
            if cred.count()==1:
                for i in cred:
                    f.user=i
                    f.feedback=req.POST['T1']
                    f.save()
            
            
            return render(req,'analysis/feedback.html')
        else:
            return render(req,'analysis/feedback.html')


    


