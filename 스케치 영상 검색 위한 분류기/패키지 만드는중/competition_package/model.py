def make_npy(image_w,image_h):

    import numpy as np
    import cv2
    from sklearn.model_selection import train_test_split
    # resize 할 이미지의 형태 

    # image_w = 28
    # image_h = 28

    # 폴더 내 카테고리
    categories = ["L2_3", "L2_10", "L2_12", "L2_15", "L2_20", "L2_21", "L2_24", "L2_25", "L2_27", "L2_30","L2_33", "L2_34", "L2_39", "L2_40", "L2_41", "L2_44", "L2_45","L2_46", "L2_50", "L2_52"]
    # 데이터 저장 공간(한국어로 하면 안됨)
    groups_folder_path = './data'

    # resize 해서 학습에 알맞은 형태로 변환
    X = []
    Y = []
    for idex, categorie in enumerate(categories):
        print(categorie)
        # label = [0 for i in range(num_classes)]
        label = idex
        image_dir = groups_folder_path  + '/' + categorie + "/"
    
        for top, dir, f in os.walk(image_dir):

            for filename in f:
                print(image_dir+filename)
                img = cv2.imread(image_dir+filename)
                img = cv2.resize(img,  (28, 28))
                X.append(img/256)
                Y.append(label)
    
    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
    xy = (X_train, X_test, Y_train, Y_test)
 
    np.save("./img_data.npy", xy)
    # 함수를 실행하면 npy 파일이 생성됨
    return 
