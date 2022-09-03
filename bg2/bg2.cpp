#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>
#include <fstream>

/// For the Raspberry Pi 32-bit Bullseye OS

std::string gstreamer_pipeline(int capture_width, int capture_height, int framerate, int display_width, int display_height) {
    return
            " libcamerasrc ! video/x-raw, "
            " width=(int)" + std::to_string(capture_width) + ","
            " height=(int)" + std::to_string(capture_height) + ","
            " framerate=(fraction)" + std::to_string(framerate) +"/1 !"
            " videoconvert ! videoscale !"
            " video/x-raw,"
            " width=(int)" + std::to_string(display_width) + ","
            " height=(int)" + std::to_string(display_height) + " ! appsink";
}

using namespace cv;
int main(int argc, char* argv[])
{
    //pipeline parameters
    int capture_width = 640; //1280 ;
    int capture_height = 480; //720 ;
    int framerate = 30 ;
    int display_width = 640; //1280 ;
    int display_height = 480; //720 ;

    //reset frame average
    std::string pipeline = gstreamer_pipeline(capture_width, capture_height, framerate, display_width, display_height);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n\n\n";
    
    VideoCapture cap(pipeline, CAP_GSTREAMER);
    if(!cap.isOpened()) {
        std::cout<<"Failed to open camera."<<std::endl;
        return (-1);
    }
    
    Ptr<BackgroundSubtractor> pBackSub;
    pBackSub = createBackgroundSubtractorMOG2();
    
    std::vector<std::vector<Point>> contours;
    
    std::vector<std::vector<Point>> positions;
    
    namedWindow("Camera", WINDOW_AUTOSIZE);
    Mat frame, background, fgMask, gray;
    
    // run the program in "write" mode, meaning we save the first frame
    if (argc > 1 && std::string(argv[1]) == "write") {
        cap.read(frame);
        flip(frame, frame, 1);
        cvtColor(frame, background, COLOR_BGR2GRAY);      
        imwrite("background.jpg",background);
        
        std::cout << "Successfully wrote background image to disk" << std::endl;
        imshow("Camera", background);
        waitKey(1000);
    }
    // run the program in "read" mode, meaning we load a saved background image
    else {
        background = imread("background.jpg");
        cvtColor(background, background, COLOR_BGR2GRAY);
        std::cout << "Successfully loaded background image from disk"<< std::endl;
        imshow("Camera", background);
        waitKey(1000);
    }
    
    
    pBackSub->apply(background, fgMask, 1); // learn the background
    
    
    std::cout << "Hit ESC to exit" << std::endl;
    int frameNum = 0;
    while(true)
    {
        std::cout << "Frame_" << frameNum++;
    	if (!cap.read(frame)) {
            std::cout<<"Capture read error"<<std::endl;
            break; 
        }
        else if (frame.empty()){
            std::cout<<"Error: frame is empty"<<std::endl;
            break;
        }
        
        flip(frame, frame, 1);
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // background subtraction
        pBackSub->apply(gray, fgMask, 0.001); // no learning
        threshold(fgMask, fgMask, 20, 255, THRESH_BINARY);
        
        // try to eliminate string
        erode(fgMask, fgMask, 500);
        
        //~ absdiff(gray, background, fgMask);
        
        // find contours
        findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        
        // Canny Edge Detection
        //~ Canny(fgMask, fgMask, 50, 200);
        
        std::vector<Point> centroids;
        
        for(int i = 0; i < contours.size(); i++) {
            
            //double peri = arcLength(contours[i], true);
            double area = contourArea(contours[i]);
            
            
            if (area > 300) {
                
                // calculate center of object
                Rect rect = boundingRect(contours[i]);
                centroids.push_back(
                    Point(rect.x+rect.width/2,rect.y+rect.height/2)
                );
                
                
                
                drawContours(frame, contours, i, Scalar(255,0,0), 2);
                rectangle(frame, rect, Scalar(0,255,0));   
            }
        }
        
        positions.push_back(centroids);
        for(int i =0; i < centroids.size(); i++) {
            std::cout << centroids[i] << " ";
        }
        std::cout << std::endl;
        
        //show frame
        imshow("Camera",frame);

        char esc = waitKey(1);
        if(esc == 27) break;
    }
    
    // write coordinates to file
    std::cout << "Writing centroids to file...";
    std::ofstream MyFile("positions.csv");
    for (int i =0; i < positions.size(); i++) {
        MyFile << "Frame_" << i;
        for (int j=0; j < positions[i].size(); j++) {
            MyFile << " - " << positions[i][j];
        }
        MyFile << std::endl;
    }
    MyFile.close();
    std::cout << "finished!" << std::endl;

    // clean up camera and windows
    cap.release();
    destroyAllWindows() ;
    return 0;
}

