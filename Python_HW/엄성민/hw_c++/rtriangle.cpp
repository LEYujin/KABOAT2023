#include <iostream>
#include <algorithm>

int main() {

	int s1, s2, s3;
    while (1) {
        std:: cout << "�� ���� �Է��ϼ���";
        std::cin >> s1 >> s2 >> s3;
        if (s1 > 0 && s2 > 0 && s3 > 0) {
            break;
        }
        else std::cout << " 0���� ū ���� �ٽ� �Է��ϼ���" << std::endl;
    }
    int sides[] = { s1, s2, s3 };
	std::sort(sides, sides + 3);

    if (sides[0] + sides[1] <= sides[2]) {
        std::cout << "NO" << std::endl;
    }
    else if (sides[0] * sides[0] + sides[1] * sides[1] == sides[2] * sides[2]) {
        std::cout << "RI" << std::endl;
    }
    else if (sides[0] * sides[0] + sides[1] * sides[1] < sides[2] * sides[2]) {
        std::cout << "OB" << std::endl; 
    }
    else {
        std::cout << "AC" << std::endl; 
    }
    return 0;



}//finish