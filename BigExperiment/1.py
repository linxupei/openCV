% % % % % % % % % % % 简单的车牌识别系统 % % % % % % % % % % %
% % % % % % % 钟培勋 % % % % % % % % % %
% % % % % % % QQ：930109175 % % % % % % % % % %

clc;
clear
all;
close
all;
% % step1
读入图片
灰度化处理并显示原图、灰度图

% 选择图片路径
[filename, pathname] = ...
uigetfile({'*.jpg';
'*.bmp';
'*,gif'}, '选择图片');
% 合成路径 + 文件名
str0 = [pathname filename];
% 读图
I = imread(str0);
% imshow(I);
% I = imread('./original pic/car7.jpg'); % imread函数读取图像文件
[y, x, z] = size(I);
if y > 400
    rate = 400 / y;
    I = imresize(I, rate);
end
% imshow(I);
% I = imread('./original pic/car7.jpg'); % imread函数读取图像文件
% 将彩色图像转换为黑白并显示
I1 = rgb2gray(I); % rgb2gray转换成灰度图
figure(1), imshow(I), title('原始彩色图像'); % figure命令同时显示两幅图像
whos
I;
figure(2), imshow(I1), title('原始黑白图像');

% Step2
图像预处理
对原始黑白图像进行开操作得到图像背景
% I1为灰度图
[m, n] = size(I1); % 测量图像尺寸参数
GreyHist = zeros(1, 256); % 预创建存放灰度出现概率的向量
for k=0:255
GreyHist(k + 1) = length(find(I1 == k)) / (m * n); % 计算每级灰度出现的概率，将其存入GreyHist中相应位置
end
figure(3),
subplot(2, 2, 2);
bar(0: 255, GreyHist, 'g') % 绘制直方图
title('拉伸前灰度直方图')
xlabel('灰度值')
ylabel('出现概率')
subplot(2, 2, 1), imshow(I1), title('拉伸前黑白图像');
% 灰度拉伸
I1 = double(I1);
ma = double(max(max(I1)));
mi = double(min(min(I1)));
I1 = (255 / (ma - mi)) * I1 - (255 * mi) / (ma - mi);
I1 = uint8(I1);
% figure(4),
subplot(2, 2, 3);
imshow(I1);
title('灰度拉伸后黑白图像');
for k=0:255
GreyHist(k + 1) = length(find(I1 == k)) / (m * n);
end
subplot(2, 2, 4);
bar(0: 255, GreyHist, 'b')
title('拉伸后的灰度直方图')
xlabel('灰度值')
ylabel('出现概率')

% 突出目标对象
SE = strel('disk', 16); % 半径为r = 15
的圆的模板
I2 = imopen(I1, SE); % 开运算
用模板SE对灰度图I1进行腐蚀，再对腐蚀后的结果进行膨胀，使外边缘圆滑
figure(4), imshow(I2);
title('背景图像'); % 输出背景图像
% 用原始图像与背景图像作减法，增强图像
I3 = imsubtract(I1, I2); % 两幅图相减
figure(5), imshow(I3);
title('增强黑白图像'); % 输出黑白图像

% Step3
取得最佳阈值，将图像二值化
fmax1 = double(max(max(I3))); % I3的最大值并输出双精度型
fmin1 = double(min(min(I3))); % I3的最小值并输出双精度型
T = (fmax1 - (fmax1 - fmin1) / 3) / 255; % 获得最佳阈值
bw22 = im2bw(I3, T); % 转换图像为二进制图像
bw2 = double(bw22);
figure(6), imshow(bw2);
title('图像二值化'); % 得到二值图像

% % % % % % % % % % % % % % % % 车牌定位模块 % % % % % % % % % % % % % % % % % % % % % % % % % %
% % 数学形态学处理进行车牌粗定位
% % 采用彩色像素点统计，行列扫描的方式实现车牌精确定位
% % 即改进的像素中线扫描法（像素统计法）

% Step4
车牌粗定位，对得到二值图像进行边缘检测和开闭操作进行数字形态学处理

grd = edge(bw2, 'canny') % 用canny算子识别强度图像中的边界
figure(7), imshow(grd);
title('Canny算子图像边缘提取'); % 输出图像边缘
bg1 = imclose(grd, strel('rectangle', [5, 19])); % 取矩形框的闭运算
figure(8), imshow(bg1);
title('图像闭运算[5,19]'); % 输出闭运算的图像
bg3 = imopen(bg1, strel('rectangle', [5, 19])); % 取矩形框的开运算
figure(9), imshow(bg3);
title('图像开运算[5,19]'); % 输出开运算的图像
bg2 = imopen(bg3, strel('rectangle', [11, 5])); % 取矩形框的开运算
% bg2 = bwareaopen(bg2, ); % 消除细小对象
figure(10), imshow(bg2);
title('图像开运算[11,5]'); % 输出开运算的图像
bg2 = bwareaopen(bg2, 5); % 消除细小对象
figure(11), imshow(bg2);
title('消除小对象');

% Step5
像素中线扫描（颜色纹理范围定义，行列扫描的方式）粗定位和经验阈值分割车牌

% % % % % % % % % % % % % % % % Y方向 % % % % % % % % % % % % % % % %
% 进一步确定y方向（水平方向）的车牌区域
[y, x, z] = size(bg2); % y方向对应行，x方向对应列，z方向对应深度，z = 1
为二值图像
myI = double(bg2); % 数据类型转换，每个方向范围在0
~1
0
为黑，1
为白（车牌区域）
Im1 = zeros(y, x); % 创建一个与图像一样大小的空矩阵，用于记录行扫描时蓝色像素点的位置
Im2 = zeros(y, x); % 创建一个与图像一样大小的空矩阵，用于记录列扫描时蓝色像素点的位置
Blue_y = zeros(y, 1); % 创建一个列向量，同于统计行扫描某行的蓝色像素点个数
% 开始行扫描，对每一个像素进行分析，统计满足条件的像素所在行对应的个数，确定车牌的上下边界
for i=1:y % 行扫描
for j=1:x
if (myI(i, j, 1) == 1) % 在RGB彩色模型中（0，0，1）表示蓝色，转换数据后 1为蓝色
Blue_y(i, 1) = Blue_y(i, 1) + 1; % 统计第i行蓝色像素点的个数
Im1(i, j) = 1; % 标记蓝色像素点的位置
end
end
end

% Y方向车牌区域确定
[temp, MaxY] = max(Blue_y);

% 阈值的设置是经验，采用统计分析方法和车牌的固定特征设置阈值，在规定大小的车辆图像上车牌区域的长宽经过统计，收敛于某个值
Th = 5; % 阈值参数可改（要提取的蓝颜色参数经验值范围）

% 向上追溯，直到车牌区域上边界
PY1 = MaxY;
while ((Blue_y(PY1, 1) >= Th) & & (PY1 > 1))
    PY1 = PY1 - 1;
end

% 向下追溯，直到车牌区域的下边界
PY2 = MaxY;
while ((Blue_y(PY2, 1) >= Th) & & (PY2 < y))
    PY2 = PY2 + 1;
end

% 对车牌区域进行校正，加框，减少车牌区域信息丢失
PY1 = PY1 - 2;
PY2 = PY2 + 2;
if PY1 < 1
    PY1 = 1;
end
if PY2 > y
    PY2 = y;
end

% 得到车牌区域
IY = I(PY1:PY2,:,:);

% % % % % % % % % X方向 % % % % % % % % % % %
% 进一步确定x方向（竖直方向）的车牌区域，确定车牌的左右边界
Blue_x = zeros(1, x); % 创建一个行向量，同于统计列扫描某行的蓝色像素点个数
% 列扫描，确定车牌的左右边界
for j=1:x
for i=PY1:PY2
if (myI(i, j, 1) == 1)
    Blue_x(1, j) = Blue_x(1, j) + 1; % 统计第j列蓝色像素点的个数
    Im2(i, j) = 1; % 标记蓝色像素点的位置
end
end
end

% 向右追溯，直到找到车牌区域左边界
PX1 = 1;
Th1 = 3; % 经验阈值的选取，可改
while (Blue_x(1, PX1) < Th1) & & (PX1 < x)
    PX1 = PX1 + 1;
end
% 向左追溯，直到找到车牌区域右边界
PX2 = x;
while (Blue_x(1, PX2) < Th1) & & (PX2 > PX1)
    PX2 = PX2 - 1;
end
% 对车牌区域进行校正，加框，减少信息丢失
PX1 = PX1 - 2;
PX2 = PX2 + 2;
if PX1 < 1
    PX1 = 1;
end
if PX2 > x
    PX2 = x;
end

% 得到车牌区域
IX = I(:, PX1: PX2,:);

% 分割车牌区域
Plate = I(PY1:PY2, PX1: PX2,:);
row = [PY1 PY2];
col = [PX1 PX2];
Im3 = Im1 + Im2; % 图像代数运算
Im3 = logical(Im3);
Im3(1: PY1,:)=0;
Im3(PY2: end,:)=0;
Im3(:, 1: PX1)=0;
Im3(:, PX2: end)=0;
% % % % % 显示 % % % % %
figure(11);
subplot(2, 2, 4);
imshow(IY);
title('行过滤结果', 'FontWeight', 'Bold');
subplot(2, 2, 2);
imshow(IX);
title('列过滤结果', 'FontWeight', 'Bold');
subplot(2, 2, 1);
imshow(I);
title('原图像', 'FontWeight', 'Bold');
subplot(2, 2, 3);
imshow(Plate);
title('车牌区域', 'FontWeight', 'Bold');
imwrite(Plate, 'Plate彩色图.jpg');
Plate1 = rgb2gray(Plate); % rgb2gray转换成灰度图
imwrite(Plate1, 'Plate灰度图.jpg');
% % Rando倾斜校正
plate = rando_bianhuan(Plate1);
plate = imrotate(Plate1, plate, 'bilinear', 'crop'); % 取值为负值向右旋转
% plate = houghbianhuan(Plate);
figure(12);
subplot(3, 1, 1);
imshow(Plate);
title('RGB车牌倾斜校正前');
subplot(3, 1, 2);
imshow(Plate1);
title('灰度车牌倾斜校正前');
subplot(3, 1, 3);
imshow(plate);
title('灰度车牌倾斜校正后');
imwrite(plate, 'Plate校正后图像.jpg');

% % % % % % % % % % % % % % % % % % % % % % % % 字符分割模块 % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % 采用垂直投影法与阈值分割、车牌固定特征分割

% Step6
对二值图像进行区域提取，并计算区域特征参数。进行区域特征参数比较，提取车牌区域
[L, num] = bwlabel(bg2, 4); % 标注二进制图像中已连接的部分
bwlabel（）return a
unmber
of
connected
object
Feastats = imfeature(L, 'basic'); % 计算图像区域的特征尺寸
L是bg2的图像矩阵
Area = [Feastats.Area]; % 区域面积
BoundingBox = [Feastats.BoundingBox]; % [x y width height]
车牌的框架大小
RGB = label2rgb(L, 'spring', 'k', 'shuffle'); % 标志图像向RGB图像转换
figure(13), imshow(RGB);
title('图像彩色标记'); % 输出框架的彩色图像

lx = 0;
for l=1:num
width = BoundingBox((l - 1) * 4 + 3); % 框架宽度的计算
hight = BoundingBox((l - 1) * 4 + 4); % 框架高度的计算
if (width > 50 & width < 130 & hight > 10 & hight < 50) % 框架的宽度和高度的范围，筛选连通域
lx = lx + 1;
Getok(lx) = l;
end
end
for k= 1:lx
l = Getok(k); % 找出符合尺寸标准的连通块，获取它的开始位置和结束位置
startcol = BoundingBox((l - 1) * 4 + 1) - 2; % 开始列
startrow = BoundingBox((l - 1) * 4 + 2) - 2; % 开始行
width = BoundingBox((l - 1) * 4 + 3) + 8; % 车牌宽
hight = BoundingBox((l - 1) * 4 + 4) + 2; % 车牌高
rato = width / hight; % 计算车牌长宽比
if rato > 2 & rato < 4
    break;
end
end
sbw1 = bw2(startrow:startrow + hight, startcol: startcol + width - 1); % 获取车牌二值子图
% % 倾斜校正
% sbw4 = rando_bianhuan(sbw1);
% sbw1 = imrotate(sbw1, sbw4, 'bilinear', 'crop');
SIM = I1(startrow:startrow + hight, startcol: startcol + width - 1); % 获取车牌灰度子图
% % 倾斜校正
% SIM4 = rando_bianhuan(SIM);
% SIM = imrotate(SIM, SIM4, 'bilinear', 'crop');

figure(14),
subplot(3, 1, 1), imshow(Plate);
title('RGB车牌子图'); % 输出车牌图像
subplot(3, 1, 2), imshow(SIM);
title('车牌灰度子图'); % 输出灰度图像
subplot(3, 1, 3), imshow(sbw1);
title('车牌二值子图'); % 输出车牌的二值图
imwrite(SIM, 'lisence.jpg');

% step7
车牌预处理
b = imread('lisence.jpg');
% b = rgb2gray(a);
imwrite(b, '1.车牌灰度图像.jpg');
figure(15);
subplot(3, 1, 1), imshow(b), title('1.车牌灰度图像')
g_max = double(max(max(b))); % 求最大灰度值并赋予双精度
g_min = double(min(min(b))); % 求最小灰度值并赋予双精度
T = round(g_max - (g_max - g_min) / 3); % T
为二值化的阈值
round()
取整
whos
b;
d = (double(b) >= T); % d: 二值图像
imwrite(d, '2.车牌二值图像.jpg');
figure(15);
subplot(3, 1, 2), imshow(d), title('2.车牌二值图像')
whos
d;
% bg4 = imopen(d, strel('rectangle', [1, 1])); % 取矩形框的开运算
bg4 = bwareaopen(d, 8); % 消除细小对象，即第二和第三之间的标点
figure(15);
subplot(3, 1, 3), imshow(bg4);
title('3.消除小对象二值车牌图像'); % 输出框架的彩色图像
imwrite(bg4, '3.消除小对象二值车牌图像.jpg');

% Step8
计算车牌水平投影，并对水平投影进行峰谷分析
histcol1 = sum(bg4); % 计算垂直投影
histrow = sum(bg4
');      %计算水平投影
figure(16), subplot(2, 1, 1), bar(histcol1);
title('垂直投影（含边框）'); % 输出垂直投影
subplot(2, 1, 2), bar(histrow);
title('水平投影（含边框）'); % 输出水平投影
figure(17), subplot(2, 1, 1), bar(histrow);
title('水平投影（含边框）'); % 输出水平投影
subplot(2, 1, 2), imshow(bg4);
title('车牌二值子图'); % 输出二值图
                   % 对水平投影进行峰谷分析
meanrow = mean(histrow); % 求水平投影的平均值
minrow = min(histrow); % 求水平投影的最小值
levelrow = (meanrow + minrow) / 2; % 把水平投影的均值和最小值的平均作为阈值
去判断是否为车牌区域
count1 = 0;
l = 1;
for k = 1:hight % hight为车牌高
if histrow(k) <= levelrow
count1 = count1 + 1; % 统计谷底点的个数，用于计算字符高度
else
if count1 >= 1
markrow(l)=k; % 上升点（点的位置）字符区域
markrow1(l)=count1; % 谷宽度（谷点的数量，即下降点至下一个上升点）
l=l+1;
end
count1=0;
end
end
markrow2=diff(markrow); % 峰距数列（上升点至下一个上升点）
[m1, n1]=size(markrow2); % m1=1, n1为总峰距数
n1=n1+1; % 峰数
markrow(l)=hight; % l指最后一个字符
markrow1(l)=count1; % 谷底点数
markrow2(n1)=markrow(l)-markrow(l-1); % 计算第l个字符的高度（包括上下边框）
l=0;
for k=1:n1 % 该循环用于找峰中心位置，即字符高度的中点
markrow3(k) = markrow(k + 1) - markrow1(k + 1); % 上一个下降点
markrow4(k) = markrow3(k) - markrow(k); % 峰宽度（上升点至下降点）
markrow5(k) = markrow3(k) - double(uint16(markrow4(k) / 2)); % 峰中心位置
end

% Step9
计算车牌旋转角度
% (1)
在上升点至下降点找第一个为1的点
[m2, n2] = size(bg4); % 车牌图像大小
[m1, n1] = size(markrow4); % 峰宽度的大小，为一个行向量
maxw = max(markrow4); % 最大宽度为字符的高度
if markrow4(1)
~ = maxw % 检测上边
ysite = 1;
k1 = 1;
for l = 1:n2
for k=1:markrow3(ysite) % 从顶边至第一个峰下降点扫描
if sbw1(k, l) == 1 % 1
为白色，即字符
xdata(k1) = l;
ydata(k1) = k;
k1 = k1 + 1;
break;
end
end
end
else % 检测下边
ysite = n1;
if markrow4(n1) == 0
    if markrow4(n1 - 1) == maxw
        ysite = 0; % 无下边
else
    ysite = n1 - 1;
end
end
if ysite ~=0
k1 = 1;
for l=1:n2
k = m2;
while k >= markrow(ysite) % 从底边至最后一个峰的上升点扫描
    if d(k, l) == 1
        xdata(k1) = l;
        ydata(k1) = k;
        k1 = k1 + 1;
        break;
    end
    k = k - 1;
end
end
end
end
% (2)
线性拟合，计算与x夹角
fresult = fit(xdata
',ydata', 'poly1'); % poly1
Y = p1 * x + p2
p1 = fresult.p1;
angle = atan(fresult.p1) * 180 / pi; % 弧度换为度，360 / 2
pi, pi = 3.14
% (3)
旋转车牌图象
subcol = imrotate(b, angle, 'bilinear', 'crop'); % 旋转车牌图象
sbw = imrotate(bg4, angle, 'bilinear', 'crop'); % 旋转图像
figure(18), subplot(2, 1, 1), imshow(b);
title('车牌灰度子图'); % 输出车牌旋转后的灰度图像标题显示车牌灰度子图
subplot(2, 1, 2), imshow(sbw);
title(''); % 输出车牌旋转后的灰度图像
title(['车牌旋转角: ', num2str(angle), '度'], 'Color', 'r'); % 显示车牌的旋转角度

% Step10
旋转车牌后重新计算车牌水平投影，去掉车牌水平边框，获取字符高度
histcol1 = sum(sbw); % 计算垂直投影
histrow = sum(sbw
'); %计算水平投影
figure(19), subplot(2, 1, 1), bar(histcol1);
title('垂直投影（旋转后）');
subplot(2, 1, 2), bar(histrow);
title('水平投影（旋转后）');
figure(20), subplot(2, 1, 1), bar(histrow);
title('水平投影（旋转后）');
subplot(2, 1, 2), imshow(sbw);
title('车牌二值子图（旋转后）');
% 去水平（上下）边框, 获取字符高度
maxhight = max(markrow2); % 获取最大峰距，即一个字符 + 一个谷底宽
findc = find(markrow2 == maxhight); % 返回的是最大峰距所在序数
rowtop = markrow(findc); % 最大峰距的上升点位置
rowbot = markrow(findc + 1) - markrow1(findc + 1); % 最大峰距的下一个上升点 - 最大峰距所在的谷底度 = 最大字符宽度下降点
sbw2 = sbw(rowtop:rowbot,:); % 子图为(rowbot - rowtop + 1)
行
分割出最大高度所在字符
maxhight = rowbot - rowtop + 1; % 字符高度(rowbot - rowtop + 1)，最大字符高度

                                                            % Step11
计算车牌垂直投影，去掉车牌垂直边框，获取车牌及字符平均宽度
histcol = sum(sbw2); % 计算垂直投影
figure(21), subplot(2, 1, 1), bar(histcol);
title('垂直投影（去水平边框后）'); % 输出车牌的垂直投影图像
subplot(2, 1, 2), imshow(sbw2); % 输出垂直投影图像
title(['车牌字符高度： ', int2str(maxhight)], 'Color', 'r'); % 输出车牌字符高度

                                                        % 对垂直投影进行峰谷分析
meancol = mean(histcol); % 求垂直投影的平均值
mincol = min(histcol); % 求垂直投影的最小值
levelcol = (meancol + mincol) / 4; % 以垂直投影的直方图平均值与最小值之和的1 / 4
为阈值，判断字符与间隔
count1 = 0;
l = 1;
for k = 1:width % width为车牌的长度
if histcol(k) <= levelcol % 小于阈值为字符间隔区域
count1 = count1 + 1;
else
if count1 >= 1
markcol(l)=k; % 字符上升点
markcol1(l)=count1; % 谷宽度（下降点至下一个上升点）
l=l+1;
end
count1=0;
end
end
markcol2=diff(markcol); % 峰距离（上升点至下一个上升点），包含一个字符宽度和一个谷底宽度
[m1, n1]=size(markcol2); % 峰距数列
n1=n1+1;
markcol(l)=width; % 把最后一个上升点设为车牌宽度右边界所在点
markcol1(l)=count1; % 最后谷底宽度
markcol2(n1)=markcol(l)-markcol(l-1);

% 对垂直投影进行峰谷分析
meancol=mean(histcol); % 求垂直投影的平均值
mincol=min(histcol); % 求垂直投影的最小值
levelcol=(meancol+mincol) / 4; % 以垂直投影的直方图平均值与最小值之和的1 / 4为阈值，判断字符与间隔
count1=0;
l=1;
for k=1:width
if histcol(k) <= levelcol
count1 = count1 + 1;
else
if count1 >= 1
markcol(l)=k; % 字符上升点
markcol1(l)=count1; % 谷宽度（下降点至下一个上升点）
l=l+1;
end
count1=0;
end
end
markcol2=diff(markcol); % 峰距离（上升点至下一个上升点）
[m1, n1]=size(markcol2); % 峰距离数列
n1=n1+1;
markcol(l)=width;
markcol1(l)=count1;
markcol2(n1)=markcol(l)-markcol(l-1);

% Step12 计算车牌上每个字符中心位置，计算最大字符宽度maxwidth
l=0;
for k=1:n1 - 1
markcol3(k) = markcol(k + 1) - markcol1(k + 1); % 第k个下降点（第k个字符的结束位置）
markcol4(k) = markcol3(k) - markcol(k); % 字符宽度（上升点至下降点）
markcol5(k) = markcol3(k) - double(uint16(markcol4(k) / 2)); % 第k个峰（字符）中心位置
end
markcol6 = diff(markcol5); % 字符中心距离数列（字符中心点至下一个字符中心点）
maxs = max(markcol6); % 查找最大值，即为第二字符与第三字符中心距离
findmax = find(markcol6 == maxs); % 获取最大值序数
markcol6(findmax) = 0; % 把第二字符与第三字符中心距离的中心距置为0
maxwidth = max(markcol6); % 继续查找最大值，即为最大字符分割宽度

                                    % Step13
提取分割字符, 并变换为40行 * 20
列标准子图
l = 1;
[m2, n2] = size(subcol);
figure(22);
for k = findmax - 1:findmax + 5
cleft = markcol5(k) - maxwidth / 2; % markcol5是第k个字符中心位置，celft为上一个谷底的平均值，即第k个字符的最佳分割左阈值
cright = markcol5(k) + maxwidth / 2 - 2; % cright为第k个字符的最佳分割右阈值
if cleft < 1
cleft = 1;
cright = maxwidth; % 确定第一个字符的分割阈值
end
if cright > n2
cright = n2; % 确定最后一个字符的分割阈值
cleft = n2 - maxwidth;
end
SegGray = sbw(rowtop:rowbot, cleft: cright);
SegBw1 = sbw(rowtop:rowbot, cleft: cright);
SegBw2 = imresize(SegBw1, [40 20]); % 变换为40行 * 20
列标准子图
subplot(2, n1 + 1, l), imshow(SegGray);
% subplot(2, n1 + 1, l), imshow(SegGray);
if l == 4
title(['车牌字符宽度： ', int2str(maxwidth)], 'Color', 'r');
end
subplot(2, n1, n1 + l), imshow(SegBw2);
% title('标准化为20*40车牌字符'], 'Color', 'r');
if l == 4
    title(['标准化为40*20车牌字符'], 'Color', 'b');
end
fname = strcat('./word/', int2str(k), '.jpg');
imwrite(SegBw2, fname, 'jpg')
l = l + 1;
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % 字符识别模块 % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% step13
用模版匹配法识别车牌字符
% % 具体步骤：
% % 建立模版库，遍历模版库与待识别字符比对
% % 计算待识别字符的与模版字符的特征向量距离，取最小值，认为是最相似
% % 比对结果组织成字符串输出

% % 逐一读入切割好的车牌字符
str = './word/';
% str1 = 'word';
for i=1:7
im = imread([str, num2str(i), '.jpg'])
% h = fspecial('motion', 4, 25);
% im = imfilter(im, h, 'replicate');
% h = fspecial('disk', 1.2);
% im = imfilter(im, h, 'replicate');
% h = fspecial('unsharp');
% im = imfilter(im, h, 'replicate');
% im = medfilt2(im);
imwrite(im, strcat('word', num2str(i), '.jpg'));
end
word1 = imread('word1.jpg');
word2 = imread('word2.jpg');
word3 = imread('word3.jpg');
word4 = imread('word4.jpg');
word5 = imread('word5.jpg');
word6 = imread('word6.jpg');
word7 = imread('word7.jpg');

% 对字符图像进行预处理、归一化
wid = [size(word1, 2) size(word2, 2) size(word3, 2) size(word4, 2)...
    size(word5, 2) size(word6, 2) size(word7, 2)];
[maxwid, indmax] = max(wid);
maxwid = maxwid + 10;
wordi = word1;
wordi = [zeros(size(wordi, 1), round((maxwid - size(word1, 2)) / 2)) wordi
         zeros(size(wordi, 1), round((maxwid - size(word1, 2)) / 2))];
word1 = wordi;
wordi = word2;
wordi = [zeros(size(wordi, 1), round((maxwid - size(word2, 2)) / 2)) wordi
         zeros(size(wordi, 1), round((maxwid - size(word2, 2)) / 2))];
word2 = wordi;
wordi = word3;
wordi = [zeros(size(wordi, 1), round((maxwid - size(word3, 2)) / 2)) wordi
         zeros(size(wordi, 1), round((maxwid - size(word3, 2)) / 2))];
word3 = wordi;
wordi = word4;
wordi = [zeros(size(wordi, 1), round((maxwid - size(word4, 2)) / 2)) wordi
         zeros(size(wordi, 1), round((maxwid - size(word4, 2)) / 2))];
word4 = wordi;
wordi = word5;
wordi = [zeros(size(wordi, 1), round((maxwid - size(word5, 2)) / 2)) wordi
         zeros(size(wordi, 1), round((maxwid - size(word5, 2)) / 2))];
word5 = wordi;
wordi = word6;
wordi = [zeros(size(wordi, 1), round((maxwid - size(word6, 2)) / 2)) wordi
         zeros(size(wordi, 1), round((maxwid - size(word6, 2)) / 2))];
word6 = wordi;
wordi = word7;
wordi = [zeros(size(wordi, 1), round((maxwid - size(word7, 2)) / 2)) wordi
         zeros(size(wordi, 1), round((maxwid - size(word7, 2)) / 2))];
word7 = wordi;
% 字符归一化大小为
40 * 20
word11 = im2bw(imresize(word1, [40 20]));
word21 = im2bw(imresize(word2, [40 20]));
word31 = im2bw(imresize(word3, [40 20]));
word41 = im2bw(imresize(word4, [40 20]));
word51 = im2bw(imresize(word5, [40 20]));
word61 = im2bw(imresize(word6, [40 20]));
word71 = im2bw(imresize(word7, [40 20]));
% % 显示归一化后字符
figure(23),
subplot(1, 7, 1);
imshow(word11);
title('1');
subplot(1, 7, 2);
imshow(word21);
title('2');
subplot(1, 7, 3);
imshow(word31);
title('3');
subplot(1, 7, 4);
imshow(word41);
title('4');
xlabel({'加框及归一化大小后车牌字符'}, 'Color', 'r');
subplot(1, 7, 5);
imshow(word51);
title('5');
subplot(1, 7, 6);
imshow(word61);
title('6');
subplot(1, 7, 7);
imshow(word71);
title('7');
imwrite(word11, 'word11.bmp');
imwrite(word21, 'word21.bmp');
imwrite(word31, 'word31.bmp');
imwrite(word41, 'word41.bmp');
imwrite(word51, 'word51.bmp');
imwrite(word61, 'word61.bmp');
imwrite(word71, 'word71.bmp');

% % 计算两图像欧式距离 % '标准库/*.bmp'
pattern = [];
dirpath = fullfile(pwd, '标准库/*.bmp');
files = ls(dirpath);
for t = 1: length(files)
filenamet = fullfile(pwd, '标准库', files(t,:)); % % fileparts()
函数用法
[pathstr, name, ext, versn] = fileparts(filenamet); % % file = '\home\user4\matlab\classpath.txt';
imagedata = imread(filenamet); % % [pathstr, name, ext, versn] = fileparts(file)
imagedata = im2bw(imagedata, 0.5); % 二值化 % % pathstr =\home\user4\matlab
% h = fspecial('motion', 3, 25); % % name = classpath
ext =.txt
versn = ''
% imagedata = imfilter(imagedata, h, 'replicate');
% h = fspecial('disk', 1.8);
% imagedata = imfilter(imagedata, h, 'replicate');
% h = fspecial('unsharp');
% imagedata = imfilter(imagedata, h, 'replicate');
% imagedata = medfilt2(imagedata); % 中值滤波
pattern(t).feature = imagedata;
pattern(t).name = name; % 取模版字符名字
end

distance = [];
for m = 1: 7;
for n = 1: length(files);
switch
m
case
1
distance(n) = sum(sum(abs(word11 - pattern(n).feature)));
case
2
distance(n) = sum(sum(abs(word21 - pattern(n).feature)));
case
3
distance(n) = sum(sum(abs(word31 - pattern(n).feature)));
case
4
distance(n) = sum(sum(abs(word41 - pattern(n).feature)));
case
5
distance(n) = sum(sum(abs(word51 - pattern(n).feature)));
case
6
distance(n) = sum(sum(abs(word61 - pattern(n).feature)));
case
7
distance(n) = sum(sum(abs(word71 - pattern(n).feature)));
end
end
[yvalue, xnumber] = min(distance);
filename = files(xnumber,:);
[pathstr, name, ext] = fileparts(filename);
result(m) = name;
end
str = ['识别结果为：' result];
str = result;
figure(24),
% xlabel({'第一步：车牌定位'});
subplot(6, 7, 1: 14), imshow(Plate), title('第一步：车牌定位'),
xlabel({'第二步：车牌分割'});
subplot(6, 7, 15), imshow(word1); % title('1');
subplot(6, 7, 16), imshow(word2); % title('2');
subplot(6, 7, 17), imshow(word3); % title('3');
subplot(6, 7, 18), imshow(word4); % title('4');
subplot(6, 7, 19), imshow(word5); % title('5');
subplot(6, 7, 20), imshow(word6); % title('6');
subplot(6, 7, 21), imshow(word7); % title('7');
subplot(6, 7, 26: 38), imshow(bg4);
title('车牌二值图');
xlabel(['第三步：识别结果为(已存入excel表格): ', str], 'Color', 'r');
msgbox(str, '车牌识别', 'modal');
% 导出文本到excel
fid = fopen('车牌登记.xls', 'a+');
fprintf(fid, '%s\r\n', str, datestr(now));
fclose(fid);



