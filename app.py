import streamlit as st
import pickle
import os
import pandas as pd
import xgboost as xgb


    
def new_data(goal, assist, apperance, prob, g_prob, foul_p, off_p, y_p, r_p, c_p):
    with open('km.model', 'rb') as f:
        model = pickle.load(f)
    nw = pd.DataFrame([{'득점' : goal, '도움' : assist, '출장' : apperance, '경기당 기록' : prob, '골 전환율' : g_prob, '파울P' : 80 - foul_p, '오프사이드P' : 80 - off_p, '경고P' : 80 - y_p, '퇴장P' : 80 - r_p, '교체P' : 80 - c_p}])
    prediction = model.predict(nw)
    return prediction
    
    
def main():
    st.title('공격수 티어 분류')
    st.write('K리그 데이터 기반')
    goal = st.number_input('득점', min_value = 0)
    assist = st.number_input('도움', min_value = 0)
    apperance = st.number_input('출장', min_value = 0)
    prob = st.number_input('경기당 기록', min_value = 0.0)
    g_prob = st.number_input('골 전환율', min_value = 0.0)
    foul_p = st.number_input('파울', min_value = 0)
    off_p = st.number_input('오프사이드', min_value = 0)
    y_p = st.number_input('경고', min_value = 0)
    r_p = st.number_input('퇴장', min_value = 0)
    c_p = st.number_input('교체', min_value = 0)
    
    result_list = ['C', 'B', 'F', 'A', 'D']
    
    if st.button('분류 시작'):
        r = new_data(goal, assist, apperance, prob, g_prob, foul_p, off_p, y_p, r_p, c_p)
        for i in range(0, 5):
            if r == i:
                result = result_list[i]
        st.success(f'티어 : {result}')

if __name__ == "__main__":
    main()