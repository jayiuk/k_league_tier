import streamlit as st
import pickle
import os
import pandas as pd
import xgboost as xgb


    
def new_data(goal, assist, apperance, prob, g_prob):
    with open('xgb_tier.model', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.model', 'rb') as f:
        scaler = pickle.load(f)
    nw = pd.DataFrame([{'득점' : goal, '도움' : assist, '출장' : apperance, '경기당 기록' : prob, '골 전환율' : g_prob}])
    nws = scaler.transform(nw)
    nx = xgb.DMatrix(nws)
    prediction = model.predict(nx)
    return prediction
    
    
def main():
    st.title('공격수 티어 분류')
    st.write('K리그 데이터 기반')
    goal = st.number_input('득점', min_value = 0)
    assist = st.number_input('도움', min_value = 0)
    apperance = st.number_input('출장', min_value = 0)
    prob = st.number_input('경기당 기록', min_value = 0.0)
    g_prob = st.number_input('골 전환율', min_value = 0.0)
 
    
    result_list = ['C', 'B', 'F', 'A', 'D']
    
    if st.button('분류 시작'):
        r = new_data(goal, assist, apperance, prob, g_prob)
        for i in range(0, 5):
            if r == i:
                result = result_list[i]
        st.success(f'티어 : {result}')

if __name__ == "__main__":
    main()