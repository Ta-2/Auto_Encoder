import numpy as np
import illustrator as ill

#データの取得
X = np.loadtxt('data.csv')
row = X.shape[0]
col = 5
#print(X)

#学習率
learning_rate = 0.01
#最適化ループ上限
optimaize_loop = 1500
#最適化の閾値
threshold = 0.00001

#正規化
max_val = np.max(X, axis=0)
data = X / max_val

dim = 2
encode_W = np.random.randn(col + 1, dim)
decode_W = np.random.randn(dim + 1, col)

def forword(data, encode_W, decode_W):
    hidden_data  = np.dot(np.hstack((data, 1.0)),        encode_W)
    decoded_data = np.dot(np.hstack((hidden_data, 1.0)), decode_W)

    return (hidden_data, decoded_data)

def calc_error(data, decoded_data):
    dif   = decoded_data - data
    error = np.dot(dif, dif.T) / 2.0
    
    return error

def calc_whole_error(datas, encode_W, decode_W):
    err = 0.0
    for d in datas:
        _, decoded_data = forword(d, encode_W, decode_W)
        err += calc_error(d, decoded_data)
    
    return err

def grad_decode(data, hidden_data, decoded_data):
    dif_vec = (decoded_data - data).reshape(1, -1)
    hidden_vec = np.vstack((hidden_data.reshape(-1, 1), 1.0))
    decode_grad =  np.dot(hidden_vec, dif_vec)

    return decode_grad

def grad_encode(data, decoded_data, decode_W):
    dif_vec = (decoded_data - data).reshape(1, -1)
    data_vec = np.vstack((data.reshape(-1, 1), 1.0))
    dif_hidden_vec = np.dot(dif_vec, decode_W.T)
    encode_grad = np.dot(data_vec, dif_hidden_vec[:,0:-1])
    return encode_grad

errs = []
errs.append(calc_whole_error(data, encode_W, decode_W))
print("error value before optimizing: ")
print(errs[0])

frames = []
for i in range(optimaize_loop):
    comressed_data = []
    for d in data:
        hidden_data, decoded_data = forword(d, encode_W, decode_W)
        encode_W -= learning_rate * grad_encode(d, decoded_data, decode_W)
        decode_W -= learning_rate * grad_decode(d, hidden_data, decoded_data)
        comressed_data.append(hidden_data)
    frames.append(ill.illustrate(np.array(comressed_data), margin=1.0))
        
    now_err = calc_whole_error(data, encode_W, decode_W)
    if(np.abs(errs[-1] - now_err) < threshold):
        errs.append(now_err)
        break
    errs.append(now_err)

#アニメーションの保存
ill.seve_animation(frames, "Auto_Encoder")
ill.cla()

print("error value after optimizing: ")
print(errs[-1])

ill.line_chart(errs)
ill.show()

comressed_data = np.array([forword(d, encode_W, decode_W)[0] for d in data])
print(comressed_data)
ill.illustrate(comressed_data, margin=0.2)
ill.show()