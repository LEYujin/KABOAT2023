#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

struct Point1 {
    int x;
    int y;
};

class sakak {
public:
    Point1 p1;
    Point1 p2;
};

int main() {

    sakak paper1;
    vector<sakak> p_array;
    cout << "ù��° ���� x,y ���� �����̽��ٷ� ��� �Է��ϼ��� " << endl;
    
    string input;
    bool exitLoop = false;
    while (!exitLoop) {
        getline(cin, input);
        if (input == " ") {
            break;
        }
        else {
            istringstream iss(input);
            if (iss >> paper1.p1.x >> paper1.p1.y >> paper1.p2.x >> paper1.p2.y) {
                p_array.push_back(paper1);
            }
            else {
                break;
            }
        }
    }

    int tem_x1;
    int tem_x2;
    int tem_y1;
    int tem_y2;
    int arr_size = p_array.size();
    for (int i = 0; i < arr_size -1; i++) {
        //x �ٲٱ�
        if(p_array[i].p1.x< p_array[i+1].p1.x){
            if (p_array[i].p2.x > p_array[i + 1].p1.x) {
                tem_x1 = p_array[i + 1].p1.x;
                if (p_array[i].p2.x > p_array[i + 1].p2.x) {
                    tem_x2 = p_array[i+1].p2.x;
                } else tem_x2 = p_array[i].p2.x;

            }
            else {
                cout << "��ġ�� �ʴ´�." << endl;
                break;
            }
        }
        else if(p_array[i].p1.x > p_array[i + 1].p1.x) {
            
            if (p_array[i].p1.x < p_array[i + 1].p2.x) {
                tem_x1 = p_array[i].p1.x;

                if (p_array[i].p2.x < p_array[i + 1].p2.x) {
                    tem_x2 = p_array[i].p2.x;
                }else tem_x2 = p_array[i + 1].p2.x;

            }
            else {
                cout << "��ġ�� �ʴ´�." << endl;
                break;
            }
        }
        else {
            tem_x1 = p_array[i].p1.x;
            tem_x2 = (p_array[i].p2.x < p_array[i + 1].p2.x) ? p_array[i].p2.x : p_array[i + 1].p2.x;
        }
        //y�ٲٱ�
        if (p_array[i].p1.y < p_array[i + 1].p1.y) {
            if (p_array[i].p2.y > p_array[i + 1].p1.y) {
                tem_y1 = p_array[i + 1].p1.y;
                if (p_array[i].p2.y > p_array[i + 1].p2.y) {
                    tem_y2 = p_array[i + 1].p2.y;
                }
                else tem_y2 = p_array[i].p2.y;
            }
            else {
                cout << "��ġ�� �ʴ´�." << endl;
                break;
            }
        }
        else if (p_array[i].p1.y > p_array[i + 1].p1.y) {

            if (p_array[i].p1.y < p_array[i + 1].p2.y) {
                tem_y1 = p_array[i].p1.y;

                if (p_array[i].p2.y < p_array[i + 1].p2.y) {
                    tem_y2 = p_array[i].p2.y;
                }
                else tem_y2 = p_array[i + 1].p2.y;
            }
            else {
                cout << "��ġ�� �ʴ´�." << endl;
                break;
            }
        }
        else {
            tem_y1 = p_array[i].p1.y;
            tem_y2 = (p_array[i].p2.y < p_array[i + 1].p2.y) ? p_array[i].p2.y : p_array[i + 1].p2.y;
        }
        p_array[i + 1].p1.x = tem_x1;
        p_array[i + 1].p2.x = tem_x2;
        p_array[i + 1].p1.y = tem_y1;
        p_array[i + 1].p2.y = tem_y2;

    }
    cout << (p_array[arr_size-1].p2.x - p_array[arr_size-1].p1.x) * (p_array[arr_size-1].p2.y - p_array[arr_size-1].p1.y);
    //���������� ���� �����غ���

    return 0;
}