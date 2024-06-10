import streamlit as st
import pickle
import os
import pandas as pd
import xgboost as xgb


    
def new_data(goal, assist, foul, yellow, red, apperance, change, prob):
    with open('xgb_tier.model', 'rb') as f:
        model = pickle.load(f)
    nw = pd.DataFrame([{'득점' : goal, '도움' : assist, '파울' : foul, '경고' : yellow, '퇴장' : red, '출장' : apperance, '교체' : change, '경기당 기록' : prob}])
    nx = xgb.DMatrix(data = nw)    
    prediction = model.predict(nx)
    prediction = int(prediction)
    return prediction
    
    
def main():
    st.title('공격수 티어 분류')
    st.write('K리그 데이터 기반')
    goal = st.number_input('득점', min_value = 0)
    assist = st.number_input('도움', min_value = 0)
    foul = st.number_input('파울', min_value = 0)
    yellow = st.number_input('경고', min_value = 0)
    red = st.number_input('퇴장', min_value = 0)
    apperance = st.number_input('출장', min_value = 0)
    change = st.number_input('교체', min_value = 0)
    prob = st.number_input('경기당 기록', min_value = 0.0)
    
    result_list = ['S', 'D', 'C+', 'B', 'D+', 'C', 'A', 'A+', 'F', 'B+']
    
    if st.button('분류 시작'):
        r = new_data(goal, assist, foul, yellow, red, apperance, change, prob)
        for i in range(0, 10):
            if r == i:
                result = result_list[r]
        st.success(f'티어 : {result}')

if __name__ == "__main__":
    main()