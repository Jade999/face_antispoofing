# face_antispoofing

this code takes surf feature to detect spoofing face in recognition.

1. the linear svm was selef contained, and you still need to configure two other libraries:
    - Opencv3.0+ with opencv contrib and non-free module
    - vlfeat
2. compile the project, for example

    ```bash
    g++ live_demo.cpp linear.cpp tron.cpp `pkg-config opencv --cflags --libs` -lvl -lblas
    ```

    **note: OpenCV 3.4.3 seems to have some problem with surf module. OpenCV 3.4.2 works just fine**

3. this code is implemented using c/c++ according to my understanding to  [<< Face Anti-Spoofing using Speeded-Up Robust Features and Fisher Vector Encoding >> ](http://ieeexplore.ieee.org/document/7748511/?reload=true), thanks the authors, and you can read this paper if needed.
