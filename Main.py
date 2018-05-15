from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.Improved_system import Improved_system
from utils.scorer import report_score


def execute_demo(language):
	data = Dataset(language)

	print("{}: {} training - {} test".format(language, len(data.trainset), len(data.devset)))
	
	#baseline = Baseline(language)#, data.trainset)

	#baseline.train(data.trainset)#,data.devset)


	baseline = Improved_system(language, data.trainset)

	baseline.train(data.trainset,data.devset)
	predictions = baseline.test(data.devset)

	gold_labels = [sent['gold_label'] for sent in data.devset]

	report_score(gold_labels, predictions)

	predictions = baseline.test(data.testset)

	gold_labels = [sent['gold_label'] for sent in data.testset]

	report_score(gold_labels, predictions)
if __name__ == '__main__':
	execute_demo('english')
	execute_demo('spanish')


