from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from time import sleep
from django.urls import reverse_lazy

# Create your views here.
from django.views.generic import TemplateView

from .Backend.extract_apks import generate_description_from_apk
from .Backend.tasks import go_to_sleep


class Home(TemplateView):
    template_name = 'appUpload/index.html'

'''
def upload(request):
    task = go_to_sleep.delay(5)
    return render(request,'appUpload/generateDesc.html',{'task_id' : task.task_id})

'''
def upload(request):
    context = {}
    i=0
    if request.method == 'POST':
        i=1
        uploaded_file = request.FILES['document']
        print(uploaded_file.name.removesuffix('.apk'))
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)

    if i==0:
        task = go_to_sleep.delay(100)
    elif i==1:
        task = generate_description_from_apk.delay(uploaded_file.name.removesuffix('.apk'))
        context['outputApplicationDescription'] = task

    context['task_id'] = task.task_id
    return render(request, 'appUpload/generateDesc.html',context)
