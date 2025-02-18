"""
URL configuration for django_mnist project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from . import views, search

urlpatterns = [
    path('', views.index),
    path('cnn', views.cnn, name='cnn'),
    path('gan', views.gan, name='gan'),
    path("hello/", views.hello, name="hello"),
    path('search/', search.search, name='search'),
    path('search-form/', search.search_form, name='search_form'),
    path('cnn-process/', views.cnn_process, name='cnn_process'),
    path('gan-process/', views.gan_process, name='gan_process'),
]
