const canvas = document.getElementById('paintCanvas');
const ctx = canvas.getContext('2d');
const undoButton = document.getElementById('undoButton');
const clearButton = document.getElementById('clearButton');
const saveButton = document.getElementById('saveButton');
const resultDiv = document.getElementById('result');

let drawing = false;
let history = [];
let operatedTimestamp = 0;
let blank = true;

// 初始化画布，设置背景为白色
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / 10; // 缩放比例
    const y = (e.clientY - rect.top) / 10; // 缩放比例
    ctx.beginPath();
    ctx.moveTo(x, y);
});

canvas.addEventListener('mousemove', (e) => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / 10; // 缩放比例
    const y = (e.clientY - rect.top) / 10; // 缩放比例
    ctx.lineTo(x, y);
    ctx.stroke();
});

canvas.addEventListener('mouseup', () => {
    drawing = false;
    history.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    operatedTimestamp = Date.now();
    blank = false;
});

undoButton.addEventListener('click', () => {
    if (blank) return;
    if (history.length <= 0) return;
    if (history.length === 1) {
        history.pop();
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        blank = true;
    } else {
        history.pop();
        ctx.putImageData(history[history.length - 1], 0, 0);
    }
    operatedTimestamp = Date.now();
});

clearButton.addEventListener('click', () => {
    if (blank) return;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    history = [];
    operatedTimestamp = Date.now();
    blank = true;
});

saveButton.addEventListener('click', () => {
    const timestamp = Date.now();
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = `${timestamp}.png`;
    link.click();
});

setInterval(() => {
    if (Date.now() - operatedTimestamp > 3000) {
        return;
    }
    const imageData = canvas.toDataURL('image/png');
    fetch('/cnn-process/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: imageData,
        }),
    }).then(response => {
        return response.json();
    }).then(data => {
        console.log(data);
        resultDiv.textContent = `预测结果: ${data.predicted}`;
    }).catch(error => {
        console.error(error);
    });
}, 1000);

// 绘制条形图的函数
function drawBarChart(data, ctx, canvas) {
    // 设置条形图的参数
    const barWidth = canvas.width / data.length; // 条形的宽度
    const barSpacing = 2; // 条形之间的间距
    const maxBarHeight = canvas.height - 20; // 条形的最大高度
    const maxValue = Math.max(...data); // 数据中的最大值

    // 绘制条形图
    data.forEach((value, index) => {
        const barHeight = (value / maxValue) * maxBarHeight; // 根据数据值计算条形的高度
        const x = index * (barWidth + barSpacing); // 条形的x坐标
        const y = canvas.height - barHeight; // 条形的y坐标

        // 绘制条形
        ctx.fillStyle = `hsl(${(index / 10) * 360}, 50%, 50%)`;
        ctx.fillRect(x, y, barWidth, barHeight);

        // 绘制数值标签
        ctx.fillStyle = 'black';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(index, x + barWidth / 2, canvas.height + 5);
    });
}
