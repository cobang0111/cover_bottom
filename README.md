- Environment
    
    ```bash
    # python version -> 3.10
    
    # env file
    https://drive.google.com/file/d/1HAraIbcEMmzOzyPwQygJx6_mPIZmsD2C/view?usp=drive_link

    # model file
    https://drive.google.com/file/d/1ZJgik_-80Dio35M3Bzo3XroQ6XKG94EL/view?usp=drive_link
    
    tar -xzf cb_env.tar.gz -C envs/cover_bottom


    # Environment 활성
    source envs/cover_bottom/bin/activate

    conda-unpack
    
    # OCC test
    envs/cover_bottom/bin/python -c "import OCC; print(OCC.VERSION)"
    
    pip install -r requirements.txt 
    ```
    
- Backend
    
    ```bash
    # source envs/cover_bottom/bin/activate
    
    cd backend
    
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    
- Frontend
    
    ```bash
    # source envs/cover_bottom/bin/activate

    cd frontend
    
    # 최초 설치 시 
    npm install
    
    npm run dev
    ```
    
- 로컬 접속
    
    ```bash
     http://localhost:3000
    ```

- 선택 사항 체크 박스 삭제
    
    ```bash
     # frontend/src/app/simulation/cover_bottom/page.tsx
     # Line 796-806 주석 처리
    ```
