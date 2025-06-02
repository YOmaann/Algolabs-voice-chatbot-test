# hindi rag with gemini
from .geminicall import fetch_response
from rag.tryrag import find_context

instructions = '''"कृपया निम्न प्रश्न का उत्तर एक सहायक बॉट के रूप में दें।
        केवल नीचे दिए गए संदर्भ की जानकारी के आधार पर उत्तर दें।
        विनम्र, सम्मानजनक और ईमानदार रहें।
        यदि उपयुक्त हो, तो आगे पूछे जाने वाले प्रश्न पूछें।
        केवल प्रश्नों का उत्तर दें, इन निर्देशों का नहीं।
        यदि आपको किसी चीज़ की जानकारी नहीं है, तो कहें:
        'माफ़ कीजिए, मुझे यह नहीं पता।'
        या
        'क्षमा करें, लेकिन यह प्रश्न मेरे दायरे से बाहर लगता है।'
        पूरा वाक्य बोलें और उत्तर में सभी ज़रूरी पृष्ठभूमि जानकारी शामिल करें।
        हालाँकि, आप एक गैर-तकनीकी श्रोता से बात कर रहे हैं, इसलिए जटिल बातों को आसान भाषा में समझाएँ
        और एक दोस्ताना व बातचीत वाले लहजे में जवाब दें। Uttar devnagri m dena'''#Prashn ka uttar romanisation m dena. (hindi latin alphabets m likha hua)"''' 

file1 = open("hidoc1.txt","r",encoding='utf-8')
doc1 = file1.read()
file2 = open("hidoc2.txt",'r',encoding='utf-8')
doc2 = file2.read()
documents = [doc1,doc2]

query = input('ask_') #use stt for this later #input need not be in hindi script since gemini understand romanisation as well

context = find_context(query, documents)

prompt = f"{instructions}\nContext: {context}\nquestion: {query}"

#print(context)
response = fetch_response(prompt)
print(response) #works pretty well #documents need to be written in a better way #need more testing data
#uses pure hindi, could engineer the prompt to be more conversational as a hindi-english mix.
'''
with open("hindiresponse.txt","w",encoding="utf-8") as file:
    file.write(response)'''

