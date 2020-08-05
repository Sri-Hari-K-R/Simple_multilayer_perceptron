#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include<math.h>
using namespace std;
const int inputs = 2;
const int bias = 2;
const int hidden = 2;
const int output = 2;

long double* input_arr = (long double*)malloc(inputs * sizeof(long double));
long double* weights_ih = (long double*)malloc(inputs*hidden*sizeof(long double));
long double* weights_ho = (long double*)malloc(hidden*output*sizeof(long double));
long double* hidden_arr = (long double*)malloc(hidden * sizeof(long double));
long double* hidden_out = (long double*)malloc(hidden * sizeof(long double));
long double* output_arr = (long double*)malloc(output * sizeof(long double));
long double* output_out = (long double*)malloc(output * sizeof(long double));
long double* target_out = (long double*)malloc(output * sizeof(long double));
long double* bias_arr = (long double*)malloc(bias * sizeof(long double));
long double* error_arr = (long double*)malloc(output * sizeof(long double));
long double error_total;

long double* dweights_ih = (long double*)malloc(inputs*hidden*sizeof(long double));
long double* dweights_ho = (long double*)malloc(hidden*output*sizeof(long double));

long double lr = 0.5;

long double x;

int n =(inputs*hidden)+(hidden*output);

long double temp1,temp2,temp3,temp4;



void readInput(){
    fstream input_file;

    input_file.open("D:/at/input.txt");
    if (!input_file) {
        cout << "Unable to open file";
        exit(0); // terminate with error
    }
    for(int i = 0;i<inputs;i++){
            input_file>>x;
            input_arr[i]=x;
    }
    cout<<endl;
    input_file.close();
}
void readWeights(){
    fstream weights_file;
    weights_file.open("D:/at/weights.txt");
    if (!weights_file) {
        cout << "Unable to open file";
        exit(0); // terminate with error
    }

    for(int i=0;i<hidden;i++){
        for(int j=0;j<inputs;j++){
            weights_file>>x;
            weights_ih[i*inputs+j]=x;
        }
    }
    for(int i=0;i<output;i++){
        for(int j=0;j<hidden;j++){
            weights_file>>x;
            weights_ho[i*hidden+j]=x;
        }
    }
    
    bias_arr[0]=0.35;
    bias_arr[1]=0.6;

    weights_file.close();
}
void writeWeights(){
    fstream weights_file;
    weights_file.open("D:/at/updated_weights.txt");
    if (!weights_file) {
        cout << "Unable to open file";
        exit(0); // terminate with error
    }

    for(int i=0;i<hidden;i++){
        for(int j=0;j<inputs;j++){
            weights_file << weights_ih[i*inputs+j]<<endl;
        }
    }
    for(int i=0;i<output;i++){
        for(int j=0;j<hidden;j++){
            weights_file<< weights_ho[i*hidden+j]<<endl;
        }
    }
    weights_file.close();
}
void forwardCalc(){
    for(int i=0;i<hidden;i++){
        hidden_arr[i]=0;
        for(int j=0;j<inputs;j++){
            hidden_arr[i] += input_arr[j]*weights_ih[i*inputs+j];
        }
        hidden_arr[i]+=bias_arr[0];
        hidden_out[i]= 1/(1+exp(-hidden_arr[i]));
    }
    for(int i=0;i<output;i++){
        output_arr[i]=0;
        for(int j=0;j<hidden;j++){
            output_arr[i] += hidden_out[j]*weights_ho[i*hidden+j];
        }
        output_arr[i]+=bias_arr[1];
        output_out[i] = 1/(1+exp(-output_arr[i]));
    }
}

void totalError(){
    error_total = 0;
    for(int i=0;i<output;i++){
        error_arr[i]= pow((target_out[i]-output_out[i]),2)/2;
        error_total+=error_arr[i];
    }
}

void backwardCalc(){
    dweights_ho[0]=-(target_out[0]-output_out[0])*(output_out[0]*(1-output_out[0]))*hidden_out[0];
    dweights_ho[1]=-(target_out[0]-output_out[0])*(output_out[0]*(1-output_out[0]))*hidden_out[1];

    dweights_ho[2]=-(target_out[1]-output_out[1])*(output_out[1]*(1-output_out[1]))*hidden_out[0];
    dweights_ho[3]=-(target_out[1]-output_out[1])*(output_out[1]*(1-output_out[1]))*hidden_out[1];

    
    temp1 = -(target_out[0]-output_out[0])*(output_out[0]*(1-output_out[0]));
    temp2 = -(target_out[1]-output_out[1])*(output_out[1]*(1-output_out[1]));

    temp3 = hidden_out[0]*(1-hidden_out[0]); 
    temp4 = hidden_out[1]*(1-hidden_out[1]);
    
    dweights_ih[0] = (temp1*weights_ho[0] + temp2*weights_ho[2])*temp3*input_arr[0]; 
    dweights_ih[1] = (temp1*weights_ho[0] + temp2*weights_ho[2])*temp3*input_arr[1]; 

    dweights_ih[2]= (temp1*weights_ho[1]+temp2*weights_ho[3])*temp4*input_arr[0];
    dweights_ih[3]= (temp1*weights_ho[1]+temp2*weights_ho[3])*temp4*input_arr[1];

    for(int i=0;i<4;i++){
        weights_ho[i]-= lr*dweights_ho[i];
    }
    for(int i=0;i<4;i++){
        weights_ih[i]-= lr*dweights_ih[i];
    }
}

void displayResults(){
    cout<<endl;
    cout<<"Inputs :"<<endl;
    for(int i = 0;i<inputs;i++){
            cout<<"i"<<i+1<<"="<<setprecision(9)<<input_arr[i]<<endl;
    }
    cout<<endl;
    cout<<"Weights :"<<endl;
    for(int i = 0;i<4;i++){
        cout<<"w"<<i+1<<"="<<setprecision(9)<<weights_ih[i]<<"\t\t\tw"<<i+5<<"="<<setprecision(9)<<weights_ho[i]<<endl;
    }
    cout<<endl;
    cout<<"Hidden layer outputs : "<<endl;
    for(int i = 0;i<2;i++){
        cout<<"out_h"<<i+1<<"="<<setprecision(9)<<hidden_out[i]<<endl;
    }
    cout<<endl;
    cout<<"Outputs :"<<endl;
    for(int i=0;i<output;i++){
        cout<<"o"<<i+1<<"="<<setprecision(9)<<output_out[i]<<endl;
    }
    cout<<endl;
    cout<<"Total Error :"<<endl<<setprecision(9)<<error_total;
    cout<<endl;
}

int main(){
    target_out[0]=0.01;
    target_out[1]=0.99;
    readInput();
    readWeights();
    forwardCalc();
    totalError();
    displayResults();
    for(int j=0;j<10;j++){
        backwardCalc();
        forwardCalc();
        totalError();
    }
    displayResults();
    writeWeights();

    return 0;
}