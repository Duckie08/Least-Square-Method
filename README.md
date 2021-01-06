Bình phương tối thiểu là một dạng phân tích hồi qui toán học được sử dụng để xác định đường biểu diễn phù hợp nhất cho một tập dữ liệu, cung cấp một phép minh họa trực quan về mối quan hệ giữa các điểm dữ liệu trong tập dữ liệu. 
## Cài đặt
Muốn chạy được code thì phải cài các thư viện numpy, matplotlib,.. như sau

```python

import numpy as np 
import matplotlib.pyplot as plt
```
***
### Cách sử dụng
* Trước hết ta cần nhập hệ hàm độc lập tuyến tính, bằng cách 
sử dụng lambda như mẫu dưới đây

```python
f = [np.ones_like, lambda x:x]
```
Sẽ tương ứng với hàm y = ax + b
```python

f = [np.ones_like, lambda x:x, np.sin]
```
Sẽ tương ứng hàm y = ax + b + sinx

* Sau đó chạy code sau để thực hiện tìm hệ số cho hàm

```python
[f, Bi] = least_square(x, y, f)
```
Trong đó f trong `[f,Bi]` là giá trị của hàm xấp xỉ tại 1 điểm, `Bi` là hệ số của hệ hàm

* Hàm xấp xỉ có sai số được tính bằng công thức L2 Error như sau:

```python
loss = np.sqrt(np.mean((f(x)-y)**2))
```
* Tính toán xong thì cũng cần vẽ hình để có thể thấy trực quan hàm sấp xỉ so với hàm thực tế như nào, sử dụng đến thư viện `matplotlib`

```python
xx = np.linspace(min(x), max(x))
plt.plot(x, y, 'ro')
plt.plot(xx, f(xx), '--')
plt.show()
```
Trong đó ta `xx` là tập hợp điểm trong khoảng ta muốn vẽ, `plt.plot(x, y, 'ro')` biểu diễn hàm thực tế với chấm đỏ và `plt.plot(xx, f(xx), '--')` là hàm xấp xỉ biểu diễn bởi đường kẻ
![alt](https://jp.mathworks.com/help/examples/curvefit/win64/FitPolynomialExample_02.png)
**Nhận xét**
* Để ý rằng khi bạn xấp xỉ một hàm số từ các mốc nội suy bằng một đa thức thì bậc của đa thức càng lớn thì sai số càng nhỏ. Tuy nhiên trong thực tế điều nay không hề tốt vì gây ra hiện tượng **Overfiting**, ta có thể khắc phục điều này bằng nhiều cách như LASSO, Ridge Regression,...
![Ví dụ về overfitting](https://datascience.foundation/img/pdf_images/underfitting_and_overfitting_in_machine_learning_image.png)

## Khắc phục một số hạn chế
### Ridge Regression
* Cũng tương tự như code Least square bên trên nhưng ta bổ sung thêm `XTX = XT.dot(X) + lamda*np.ones(XT.dot(X).shape)` theo như phương pháp Ridge Regression
* Vì yêu cầu hệ hàm nhập vào phải độc lập tuyến tính, tuy nhiên nếu có lỡ tay nhập hệ phụ thuộc tuyến tính thì ta có thể dùng thêm Ridge Regression để khắc phục. 
```python
[f, Bi] = least_square_ridge(x, y, f,lamda)
```
### Gradient Decent
* Ngoài việc nhập nhầm hệ hàm như trên thì quá trình giải của phương pháp cũng rất tốn thời gian vì phải thực hiện tìm ma trận nghịch đảo, Gradient Decent là một trong nhưng phương pháp thông dụng để xử lý bài toán này vì khối lượng tính toán ít hơn, cũng như giải được xấp xỉ nghiệm
```python

Dùng gradient deccend để cải tiến thuật toán
print("x:", x, x.shape)
print("y:", y, y.shape)

 W = np.random.random((3,1))
 print("W:", W, W.shape)
 X = np.asarray((x**0, x**1, np.sin(x)))
 print("X:", X, X.shape)
 Y = np.expand_dims(y, 0)
 print("Y:", Y, Y.shape)


 for step in range(10000):
 	y_hat = X.T.dot(W).T
 	loss = (y_hat - Y)**2
 	cost = np.mean(loss)
 	theta = 2*(y_hat - Y).dot(X.T).T
 	W -= 0.001*theta
 	if step % 100 == 0:
	print("Step:", step, "- W:", W.T, "- cost:", cost)

print("Bi:", Bi.T)
```

![Mô tả thuật toán](https://machinelearningcoban.com/assets/GD/1dimg_5_0.1_-5.gif)
```
