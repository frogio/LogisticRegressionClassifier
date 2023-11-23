#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#pragma warning(disable:4996)

#define CLASS_SETOSA					1
#define CLASS_VERSICOLOR				0

#define MAX_DATA						100							// ��ü ������ ����, M
#define ALPHA							0.01						// �н���
#define EPOCH							10000						// �н� Ƚ��

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

Target * LoadData();												// iris dataset�� �޾ƿ´�.
void Training(struct Target * target, struct Model * model);
double Predict(struct Model* model, double sepelLen, double sepelWidth);
void PrintTraningResult(struct Target* target, struct Model* model);

void main() {

	struct Target * target = LoadData();
	struct Model model = {1, 1, 1};									// �н����� ���� �ʱ� ��

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
void Training(struct Target* target, struct Model* model) {			// ����ϰ����� �̿��� Training

	Model diff_vec = { 0.f, };

	for (int i = 0; i < MAX_DATA; i++) {
		
		double predVal = Predict(model, target[i].sepelLen, target[i].sepelWidth);

		double error =  predVal - target[i]._class;					// ������ ���� �������� ������ ����.
		// ���� ��ȣ ����!, �ݵ�� Predict Value - Target Value������ ���־�� ��

		diff_vec.w0 += error;									
		diff_vec.w1 += error * target[i].sepelWidth;
		diff_vec.w2 += error * target[i].sepelLen;

	}

	diff_vec.w0 /= MAX_DATA;
	diff_vec.w1 /= MAX_DATA;
	diff_vec.w2 /= MAX_DATA;
	// ������� �ս��Լ� �̺а� ���

	model->w0 -= diff_vec.w0 * ALPHA;
	model->w1 -= diff_vec.w1 * ALPHA;
	model->w2 -= diff_vec.w2 * ALPHA;
	// ��� �ϰ����� ���� ����ġ ����
	// �н����� (���� �̵�����) ���� �� ���� ����ġ ����

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
	
	fgets(buf, sizeof(buf), fp);								// csv ������ ���κ��� ���� �о�´�.

	int tIdx = 0;
	for(int i = 0; i < MAX_DATA; i++){

		fgets(buf, sizeof(buf), fp);
		
		char * data = buf;

		target[tIdx].sepelLen = atof(data);						// sepelLength ������ ����

		data = strchr(buf, ',');
		*data = ' ';
		data++;

		target[tIdx].sepelWidth = atof(data);					// sepelWidth ������ ����
		
		for(int i = 0; i < 3; i++){
			data = strchr(buf, ',');
			*data = ' ';
		}
		data++;

		target[tIdx]._class = (strstr(data, "Setosa") != NULL) ? CLASS_SETOSA : CLASS_VERSICOLOR;					// ���䰪 ������ ����
		
		tIdx++;

	}
	fclose(fp);

	return target;

}