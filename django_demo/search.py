from django.http import HttpResponse, HttpRequest
from django.shortcuts import render


# 表单
def search_form(request: HttpRequest) -> HttpResponse:
	return render(request, 'search_form.html')


# 接收请求数据
def search(request: HttpRequest) -> HttpResponse:
	request.encoding = 'utf-8'
	query = request.GET['q']
	message = '搜索结果：' + query if query else '请输入关键字'
	return HttpResponse(message)


def search_post(request: HttpRequest) -> HttpResponse:
	ctx = {}
	if request.method == 'POST':
		ctx['rtl'] = request.POST['q']
