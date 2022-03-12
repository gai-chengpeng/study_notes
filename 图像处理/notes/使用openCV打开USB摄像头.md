本人使用了2种openCV提供的方法：
```
VideoCapture capture;
    capture.open(3);           //open the default camera -1才是默认摄像机，3是我的USBCaemra的
    if (capture.isOpened())
    {
        qDebug()<<"camera open!!!";
    }
    Mat edges;
    namedWindow("edges",1);
    for(;;)
    {
        Mat frame; //定义一个Mat变量，用于存储每一帧的图像
        capture>>frame; //读取当前帧
        if (!frame.empty()) //判断当前帧是否捕捉成功 **这步很重要
        {
            imshow("edges", frame); //若当前帧捕捉成功，显示
        }
        else
        {
            qDebug()<<"can not ";
        }
        waitKey(30); //延时30毫秒

    }
```

```
IplImage* pFrame = NULL;
           //声明IplImage指针
           CvCapture* pCapture = cvCreateCameraCapture(-1);
           //获取摄像头
           //-1为默认摄像头，其他则需要填写地址；
           //函数cvCreateCameraCapture给从摄像头的视频流分配和初始化CvCapture结构。
           //目前在Windows下可使用两种接口：Video for Windows（VFW）
           //和Matrox Imaging Library（MIL）；
           //Linux下也有两种接口：V4L和FireWire（IEEE1394）。
           //释放这个结构，使用函数cvReleaseCapture。
           //返回值为一个
           // CvCapture

           cvNamedWindow("video", 1);
           //创建窗口

           while(1)//显示视屏
           {
               pFrame=cvQueryFrame( pCapture );
               // 函数cvQueryFrame从摄像头或者文件中抓取一帧，
               //然后解压并返回这一帧。
               //这个函数仅仅是函数cvGrabFrame和函数cvRetrieveFrame在一起调用的组合。
               //返回的图像不可以被用户释放或者修改。抓取后，capture被指向下一帧，
               //可用cvSetCaptureProperty调整capture到合适的帧。

               if(!pFrame)break;
               //如果PFrame为空，则跳出循环；

               cvShowImage("video",pFrame);
               //当前帧显示后

               char c=cvWaitKey(33);
               //我等待33ms

               if(c==27)break;
               //如果用户触发了按键，将按键的ASCII值给C
               //如果C为ESC（ASCII 为27）循环退出
           }
           cvReleaseCapture(&pCapture);
           //释放Capture；

           cvDestroyWindow("video");
           //销毁窗口
```

以上两种方法在填入-1时，都会弹出选择摄像机的界面，而且都可以成功打开笔记本电脑上自带的摄像头。

但是本人手头上USBCamera却不行。之后发现只有使用VideoCapture方法，填入指定id才能成功过打开摄像头。 CvCapture怎么试都不行。所以希望大家在开发的时候注意到这一点。