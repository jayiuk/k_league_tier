import streamlit as st
import pickle
import os
import pandas as pd
import xgboost as xgb


    
def new_data(goal, assist, apperance, prob, g_prob, foul_p, off_p, y_p, r_p, c_p):
    with open('xgb_tier.model', 'rb') as f:
        model = pickle.load(f)
    nw = pd.DataFrame([{'득점' : goal, '도움' : assist, '출장' : apperance, '경기당 기록' : prob, '골 전환율' : g_prob, '파울P' : foul_p, '오프사이드P' : off_p, '경고P' : y_p, '퇴장P' : r_p, '교체P' : c_p}])
    nx = xgb.DMatrix(nw)
    prediction = model.predict(nx)
    return prediction
    
    
def main():
    st.title('공격수 티어 분류')
    st.write('K리그 데이터 기반')
    goal = st.number_input('득점')
    shoot = st.number_input('슈팅 수')
    assist = st.number_input('도움', min_value = 0)
    apperance = st.number_input('출장')
    foul = st.number_input('파울', min_value = 0)
    off = st.number_input('오프사이드', min_value = 0)
    y = st.number_input('경고', min_value = 0)
    red = st.number_input('퇴장', min_value = 0)
    c = st.number_input('교체', min_value = 0)
    g = goal / shoot
    p = goal / apperance
    g_prob = round(g, 1)
    prob = round(p, 2)
    foul_p = 80 - foul
    off_p = 80 - off
    y_p = 80-y
    r_p = 80 - red
    c_p = 80 - c
    
    result_list = ['C', 'B', 'F', 'A', 'D']
    
    if st.button('분류 시작'):
        r = new_data(goal, assist, apperance, prob, g_prob, foul_p, off_p, y_p, r_p, c_p)
        for i in range(0, 5):
            if r == i:
                result = result_list[i]
        st.success(f'티어 : {result}, 슈팅 : {shoot}, 골 : {goal}, 출장 수 : {apperance}, 골 전환율 : {g_prob}')

if __name__ == "__main__":
    main()