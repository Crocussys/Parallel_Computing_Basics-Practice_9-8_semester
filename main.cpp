#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

const int NUM_THREADS = 12;

int main()
{
    CascadeClassifier face_cascade;
    if (!face_cascade.load(samples::findFile("/home/rozgor/opencv/opencv-4.9.0/data/haarcascades/haarcascade_frontalface_default.xml"))){
        cout << "File Error" << endl;
        return -1;
    }
    CascadeClassifier eye_cascade;
    if (!eye_cascade.load(samples::findFile("/home/rozgor/opencv/opencv-4.9.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"))){
        cout << "File Error" << endl;
        return -1;
    }
    CascadeClassifier smile_cascade;
    if (!smile_cascade.load(samples::findFile("/home/rozgor/opencv/opencv-4.9.0/data/haarcascades/haarcascade_smile.xml"))){
        cout << "File Error" << endl;
        return -1;
    }
    VideoCapture cap("../Visual_Pattern_Recognition-Practice_9-8_semester/ZUA.mp4");
    if(!cap.isOpened()){
        cout << "Error" << endl;
        return -1;
    }
    const double FPS = cap.get(CAP_PROP_FPS);
    VideoWriter out("../Visual_Pattern_Recognition-Practice_9-8_semester/output.mp4", cap.get(CAP_PROP_FOURCC), FPS, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

    auto begin_time = chrono::steady_clock::now();
    Mat frame, temp_mat;
    vector<Rect> faces, eyes, smiles;
    bool start = false;
// #pragma omp parallel
    {
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            GaussianBlur(frame, temp_mat, Size(0, 0), 3);
            cvtColor(temp_mat, temp_mat, COLOR_BGR2GRAY);
#pragma omp parallel sections num_threads(3)
            {
#pragma omp section
                {
                    face_cascade.detectMultiScale(temp_mat, faces, 1.1, 5);
                }
#pragma omp section
                {
                    eye_cascade.detectMultiScale(temp_mat, eyes, 1.1, 5);
                }
#pragma omp section
                {
                    smile_cascade.detectMultiScale(temp_mat, smiles, 1.9, 25);
                }
            }
            for (const auto& face: faces){
                rectangle(frame, face, Scalar(0, 255, 0), 2);
            }
            for (const auto& eye: eyes){
                Point eye_center(eye.x + eye.width / 2, eye.y + eye.height / 2);
                int radius = cvRound((eye.width + eye.height) * 0.25);
                circle(frame, eye_center, radius, Scalar(255, 0, 0), 2);
            }
            for (const auto& smile: smiles){
                rectangle(frame, smile, Scalar(0, 0, 255), 2);
            }
            imshow("Faces Detected", frame);
            out.write(frame);
            char c = (char) waitKey(FPS);
            if (c == 27) break;
            if (c == 32 || start){
                while(true){
                    char c = (char) waitKey(FPS);
                    if (c == 32) break;
                }
                start = false;
            }
        }
    }
    auto end_time = chrono::steady_clock::now();
    auto elapsed_ms = chrono::duration_cast<chrono::microseconds>(end_time - begin_time);
    cout << "Ответ получен за: " << elapsed_ms.count() << "ms" << endl;
    cap.release();
    out.release();
    destroyAllWindows();
    return 0;
}
