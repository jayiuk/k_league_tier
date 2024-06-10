import streamlit as st
import pickle
import os
from dotenv import load_dotenv


load_dotenv(verbose = True)
path = os.getenv('url')
model_path = os.path.join(path, 'tier_model.pkl')
    
def new_data(model_path, goal, assist, point, shoot, apperance, change):
    model = pickle.load(model_path)
    nw = pd.DataFrame([{'골' : goal, '도움' : assist, '공격포인트' : point, '슈팅' : shoot, '출장' : apperance, '교체' : change}])
    
    prediction = model.predict(nw)
    return prediction[0]
    
    
def main():
    st.title('공격수 티어 분류')
    st.write('K리그 데이터 기반')
    goal = st.number_input('골', min_value = 0)
    assist = st.number_input('도움', min_value = 0)
    point = st.number_input('공격포인트', min_value = 0)
    shoot = st.number_input('슈팅', min_value = 0)
    apperance = st.number_input('출장', min_value = 0)
    change = st.number_input('교체', min_value = 0)
    
    if st.button('분류 시작'):
        result = new_data(model_path, goal, assist, point, shoot, apperance, change)
        if result == 0:
            result = 'B+'
        elif result == 1:
            result = 'F'
        elif result == 2:
            result = 'C+'
        elif result == 3:
            result = 'S'
        elif result == 4:
            result = 'D+'
        elif result == 5:
            result = 'A+'
        elif result == 6:
            result = 'D'
        elif result == 7:
            result = 'C'
        elif result == 8:
            result = 'B'
        elif result == 9:
            result = 'A'
        st.success(f'티어 : {result}')

if __name__ == "__main__":
    main()