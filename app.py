# Forecasting Project on user provided datasets
#Importing necessary libraries
import streamlit as st # for making web API
import pandas as pd #For data manipulation
from fbprophet import Prophet #For forecasting
from PIL import Image #For Image 

def main():
    st.title("Our First Project: Dataset Forecasting ")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Dataset Forecasting</h2>
    </div>
    """
    st.markdown("By:-")
    st.markdown("Arpit Saxena")
    st.markdown("Damanpreet Kaur")
    st.markdown("Gurnoor Singh")
    st.markdown("Harleen Kaur") 
    image = Image.open("forecasting1-removebg-preview.png") # For loading image 
    st.image(image,width=None,caption="Dataset Forecasting",use_column_width=True)
    st.set_option("deprecation.showfileUploaderEncoding", value=False) 
    data = st.file_uploader("Import the time series csv file for forecasting below ")
    # Data Preprocessing    
    if data is not None:
        df = pd.read_csv(data) # For reading csv files
        df["ds"] = pd.to_datetime(df["ds"],errors="coerce")# Converting data into actual datetime format
        st.write(df) 
        maximum_date = df["ds"].max()
        st.write("Most recent date is given below")
        st.write(maximum_date) # To display the most recent date 
        periods_input = st.slider("How many periods(or days) you would like to forecast in future? ",min_value=1,max_value=365)
    if data is not None:
        model = Prophet() 
        model.fit(df) #Training our model
        with st.spinner("Loading Please Wait...."):
            if data is not None:
                future = model.make_future_dataframe(periods=periods_input) 
                forecast_prediction=model.predict(future) 
                fcst = forecast_prediction[["ds","yhat","yhat_lower", "yhat_upper"]] 
                filtered = fcst[fcst["ds"]>maximum_date]
                st.write(filtered) 
                # Data Visualization
                fig = model.plot(forecast_prediction)
                st.write(fig)
                st.write("     The above scatterplot graph is having positive correlation , the points are scattered randomply but there is a positive relation ") 
                fig2=model.plot_components(forecast_prediction) 
                st.write(fig2)
                st.write("     In the above two  graphs , Graph 1 is in continous linear motion. Graph 2 is in non uniform motion but you can see it is having high peak near july 1  ")
    
if __name__ == "__main__":
    main()

