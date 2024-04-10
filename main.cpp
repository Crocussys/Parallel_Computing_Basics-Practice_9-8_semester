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

    cout << "0" << endl;
    auto begin_time = chrono::steady_clock::now();
    Mat frame;
    vector<Mat> frames, outputs;
    int i = 0;
    for (cap >> frame; !frame.empty(); cap >> frame){
        frames.push_back(frame.clone());
        GaussianBlur(frame, frame, Size(0, 0), 3);
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        outputs.push_back(frame.clone());
        i++;
    }
    cap.release();
    cout << "1" << endl;
    vector<vector<Rect>> faces(i), eyes(i), smiles(i);
#pragma omp parallel for num_threads(NUM_THREADS / 3)
    for (int j = 0; j < i; j++){
#pragma omp parallel sections num_threads(3)
        {
#pragma omp section
            {
                face_cascade.detectMultiScale(outputs[j], faces[j], 1.1, 5);
            }
#pragma omp section
            {
                eye_cascade.detectMultiScale(outputs[j], eyes[j], 1.1, 5);
            }
#pragma omp section
            {
                smile_cascade.detectMultiScale(outputs[j], smiles[j], 1.9, 25);
            }
        }
    }
    cout << "2" << endl;
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int j = 0; j < i; j++){
        for (const auto& face: faces[j]){
            rectangle(frames[j], face, Scalar(0, 255, 0), 2);
        }
        for (const auto& eye: eyes[j]){
            Point eye_center(eye.x + eye.width / 2, eye.y + eye.height / 2);
            int radius = cvRound((eye.width + eye.height) * 0.25);
            circle(frames[j], eye_center, radius, Scalar(255, 0, 0), 2);
        }
        for (const auto& smile: smiles[j]){
            rectangle(frames[j], smile, Scalar(0, 0, 255), 2);
        }
    }
    auto end_time = chrono::steady_clock::now();
    cout << "3" << endl;
    bool start = false;
    for (const auto& new_frame: frames){
        imshow("Faces Detected", new_frame);
        out.write(new_frame);
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
    cout << "4" << endl;
    auto elapsed_ms = chrono::duration_cast<chrono::microseconds>(end_time - begin_time);
    cout << "Ответ получен за: " << elapsed_ms.count() << "ms" << endl;
    out.release();
    destroyAllWindows();
    return 0;
}
