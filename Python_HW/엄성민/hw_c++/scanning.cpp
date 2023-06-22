#include <iostream>

#include <string>

using namespace std;


void swapCharacters(string& str, int index1, int index2) {
    char temp = str[index1];
    str[index1] = str[index2];
    str[index2] = temp;
}

bool isVowel(char c) {
    // ���� ����
    string vowels = "aeiouAEIOU";

    // c�� �������� Ȯ��
    if (vowels.find(tolower(c)) != string::npos) {
        return true;
    }
    else {
        return false;
    }
}
void scan_a(string& word, string& copy, int word_length, int count) {
    for (int i = 0; i < word_length - 1; i++) {
        swapCharacters(word, i, i + 1);
        if (isVowel(word[i])) {
            //word[i] = '*';

        }
        else if(count == 2 && word[i]=='#') {
                word[i] = copy[i];
        }else{
            word[i] = '*';
        }
     
        cout << word << endl;
    }
}

void scan_b(string& word, string& copy, int word_length, int count) {
    for (int i = word_length - 1; i > 0; i--) {
        swapCharacters(word, i, i - 1);
        if (isVowel(word[i])) {
            word[i] = '#';

        }
        else {
            //word[i] = '$';  
        }
        if (count == 2) {
            if (word[i] == '*') {
                word[i] = copy[i];
            }
        }
        cout << word << endl;
    }
}

bool containsNumber(const string& str) {
    for (char c : str) {
        if (isdigit(c)) {
            return true;
        }
    }
    return false;
}

int main() {

    string word;
    
    int num;
    int count = 1;// count�� 2���? �߰� ���


    cout << "���ڸ� �Է��ϼ���(�翷 �� �ϳ��� |)" << endl;
    while (1) {
        std::cin >> word;
        if (containsNumber(word)) {
            cout << "���ڿ��� �Է��ϼ���" << endl;
        }
        else {
            break;
        }
    }
    int word_length = word.size();
    string word_copy = word;
    // ���ʿ��� ������
    if (word[0] == '|' && word[word_length]!='|') {
        scan_a(word, word_copy, word_length, count);
        count++;
        scan_b(word, word_copy, word_length, count);
    }
    else if (word[0] != '|' && word[word_length - 1] == '|') {
        scan_b(word, word_copy, word_length, count);
        count++;
        scan_a(word, word_copy, word_length, count);
    }
    else cout << "�糡�� |�� ���ų� �Ѵ� �־�� ^____^";



    return 0;
}//finish
