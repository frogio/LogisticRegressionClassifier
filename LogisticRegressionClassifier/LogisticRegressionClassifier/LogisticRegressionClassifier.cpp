#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#pragma warning(disable:4996)

#define CLASS_SETOSA					1
#define CLASS_VERSICOLOR				0

#define MAX_DATA						100							// 전체 데이터 개수, M
#define ALPHA							0.01						// 학습률
#define EPOCH							10000						// 학습 횟수

struct Model {
	double w0;
	double w1;
	double w2;
};

struct Target {
	double sepelLen;
	double sepelWidth;
	int _class;
};

Target * LoadData();												// iris dataset을 받아온다.
void Training(struct Target * target, struct Model * model);
double Predict(struct Model* model, double sepelLen, double sepelWidth);
void PrintTraningResult(struct Target* target, struct Model* model);

void main() {

	struct Target * target = LoadData();
	struct Model model = {1, 1, 1};									// 학습되지 않은 초기 모델

	printf("Loaded Data...\n\n");
	for (int i = 0; i < MAX_DATA; i++)
		printf("sepelLength, sepelWidth, class: %lf, %lf, %d\n", target[i].sepelLen, target[i].sepelWidth, target[i]._class);
	
	for (int i = 0; i < EPOCH; i++){
	
		if (i % 1000 == 0)
			PrintTraningResult(target, &model);

		Training(target, &model);

	}

	printf("Training Result : \n");
	printf("y = %lf * x2 + %lf * x1 + %lf", model.w2, model.w1, model.w0);

	double sepelLen, sepelWidth;
	while (1) {
		printf("\nEnter sepelLength, sepelWidth (exit -1, -1): ");
		scanf("%lf,%lf", &sepelLen, &sepelWidth);
		printf("Predict Result : %s ", (Predict(&model, sepelLen, sepelWidth) > 0.5) ? "Setosa" : "Versicolor");
		
		if (sepelLen == -1)
			break;

	}

	free(target);
}

double Predict(struct Model* model, double sepelLen, double sepelWidth) {
	//		// 1, rm, lstat, dis, crim
	double u = model->w2 * sepelLen + model->w1 * sepelWidth + model->w0 * 1;

	return 1.0 / (1 + exp(-u));
}
void Training(struct Target* target, struct Model* model) {			// 경사하강법을 이용한 Training

	Model diff_vec = { 0.f, };

	for (int i = 0; i < MAX_DATA; i++) {
		
		double predVal = Predict(model, target[i].sepelLen, target[i].sepelWidth);

		double error =  predVal - target[i]._class;					// 예측한 값과 실제값의 오차를 구함.
		// 오차 부호 주의!, 반드시 Predict Value - Target Value순서로 빼주어야 함

		diff_vec.w0 += error;									
		diff_vec.w1 += error * target[i].sepelWidth;
		diff_vec.w2 += error * target[i].sepelLen;

	}

	diff_vec.w0 /= MAX_DATA;
	diff_vec.w1 /= MAX_DATA;
	diff_vec.w2 /= MAX_DATA;
	// 여기까지 손실함수 미분값 계산

	model->w0 -= diff_vec.w0 * ALPHA;
	model->w1 -= diff_vec.w1 * ALPHA;
	model->w2 -= diff_vec.w2 * ALPHA;
	// 경사 하강법을 통한 가중치 조정
	// 학습률을 (벡터 이동방향) 곱한 후 더해 가중치 조정

}

void PrintTraningResult(struct Target* target, struct Model * model) {

	double LossRate = 0;

	for (int i = 0; i < MAX_DATA; i++){
		double predicted = Predict(model, target[i].sepelLen, target[i].sepelWidth);
		LossRate += target[i]._class * log(predicted) + (1 - target[i]._class) * log(1 - predicted);
	}

	printf("Loss Rate : %lf\n", LossRate / -MAX_DATA);

}

Target * LoadData() {
	
	char buf[200] = { 0, };
	struct Target * target;

	target = (struct Target *)malloc(sizeof(Target) * MAX_DATA);

	FILE* fp = fopen("iris data.csv", "rt");
	
	fgets(buf, sizeof(buf), fp);								// csv 파일의 헤드부분을 먼저 읽어온다.

	int tIdx = 0;
	for(int i = 0; i < MAX_DATA; i++){

		fgets(buf, sizeof(buf), fp);
		
		char * data = buf;

		target[tIdx].sepelLen = atof(data);						// sepelLength 데이터 추출

		data = strchr(buf, ',');
		*data = ' ';
		data++;

		target[tIdx].sepelWidth = atof(data);					// sepelWidth 데이터 추출
		
		for(int i = 0; i < 3; i++){
			data = strchr(buf, ',');
			*data = ' ';
		}
		data++;

		target[tIdx]._class = (strstr(data, "Setosa") != NULL) ? CLASS_SETOSA : CLASS_VERSICOLOR;					// 정답값 데이터 추출
		
		tIdx++;

	}
	fclose(fp);

	return target;

}