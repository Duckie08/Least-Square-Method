# Least-Square-Method
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
* Bạn đọc có thể tìm hiểu thêm về những cái mình vừa đề cập tại một số link dưới:
* (Overfitting)[https://en.wikipedia.org/wiki/Overfitting]
## Khắc phục một số hạn chế
### Ridge Regression
* Vì yêu cầu hệ hàm nhập vào phải độc lập tuyến tính, tuy nhiên nếu có lỡ tay nhập hệ phụ thuộc tuyến tính thì ta có thể dùng thêm Ridge Regression để khắc phục. 
```python
[f, Bi] = least_square_ridge(x, y, f,lamda)
```
