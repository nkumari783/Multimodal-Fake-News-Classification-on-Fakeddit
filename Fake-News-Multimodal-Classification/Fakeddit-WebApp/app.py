import streamlit as st 
from Model import BERTResNetClassifier
import torch
from transformers import BertModel, BertTokenizer
from torchvision.transforms import v2
from PIL import Image

def get_bert_embedding(text):
        # Tokenize input text and get token IDs and attention mask
        inputs = tokenizer.encode_plus(text, add_special_tokens = True, return_tensors='pt', max_length=80, truncation=True, padding='max_length')

        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)

st.title("Fakeddit App")

uploaded_title = st.text_input("Article Headline","Lorem Ipsum") 
uploaded_file = st.file_uploader("Choose an accompanying image for the article...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Image Preview", use_container_width=True)

if st.button("Predict"): 
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name, output_hidden_states = True)
    bert_model.eval()

    if uploaded_file is not None:
        img_size = 256

        # Using the pre-calculated ImageNet mean and std values for normalization
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        transform_func = v2.Compose(
                [   v2.Resize((256,256)),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean, std)
                    ])

        input_ids, attention_mask = get_bert_embedding(uploaded_title)
        input_image = Image.open(uploaded_file)
        img_tensor = transform_func(input_image).unsqueeze(0)

        model = BERTResNetClassifier()
        model.load_state_dict(torch.load("TextImagelatest.pth", map_location=torch.device('cpu')))
        # set the model to evaluation mode
        model.eval()

        # perform inference
        #print("Input shapes:", img_tensor.shape, input_ids.shape, attention_mask.shape)
        output_text = model(
            image=img_tensor,
            text_input_ids=input_ids.unsqueeze(0),
            text_attention_mask=attention_mask.unsqueeze(0)
        )

        class_labels = [
            "TRUE", 
            "SATIRE", 
            "FALSE CONNECTION", 
            "IMPOSTER CONTENT", 
            "MANIPULATED CONTENT", 
            "MISLEADING CONTENT"
        ]
        
        predicted_class = torch.argmax(output_text, dim=1).item()
        predicted_label = class_labels[predicted_class]
        st.subheader("Predicted Category")
        st.text(f"{predicted_label}")
        if(predicted_class == 0):
             st.text("Content that is factually accurate and based on verified information.")
        if(predicted_class == 1):
             st.text("Content created for humorous or satirical purposes, often not meant to be taken seriously.")
        if(predicted_class == 2):
             st.text("Headlines or visuals that mislead or do not accurately reflect the content of the article.")
        if(predicted_class == 3):
             st.text("Content that impersonates genuine sources, such as fake news articles mimicking legitimate media outlets.")
        if(predicted_class == 4):
             st.text("Content that has been altered, edited, or manipulated to mislead or distort facts.")
        if(predicted_class == 5):
             st.text("Content that uses selective information or framing to mislead the audience, often by omitting context.")
    else:
        st.text("File Not Found")



