from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pdb

class Classifier():
	def parse_examples(self, file_name):
		examples = []
		with open(file_name, 'rb') as f:
			for line_number, line in enumerate(f):
				example = line.strip().split(",")
				print line_number
				examples.append(example)
		return examples

	def separate_features_and_labels(self, examples):
		example_ids = []
		click_labels = []
		feature_vectors = []
		for example in examples:
			[example_id, label, hour, c1, banner_pos, site_id, site_domain, site_category, app_id,
			app_domain, app_category, device_id, device_ip, device_model, device_type, device_conn_type,
			c14,c15,c16,c17,c18,c19,c20,c21] = example
			if example_id == 'id':
				continue
			example_ids.append(example_id)
			click_labels.append(label)
			feature_vector = (hour, c1, banner_pos, self._encoder(site_id), self._encoder(site_domain), self._encoder(site_category),
				self._encoder(app_id), self._encoder(app_domain), self._encoder(app_category), self._encoder(device_id), self._encoder(device_ip),
				self._encoder(device_model), device_type, device_conn_type, c14,c15,c16,c17,c18,c19,c20,c21)
			feature_vectors.append(feature_vector)
		return (example_ids, click_labels, feature_vectors)

	def get_knn_classifier(self, train_examples):
		knn = KNeighborsClassifier()
		example_ids, click_labels, feature_vectors = self.separate_features_and_labels(train_examples)
		
		knn.fit(feature_vectors, click_labels)
		
		return knn

	def classify_with_knn(self, knn_classifier, test_examples):
		example_ids, click_labels, feature_vectors = self.separate_features_and_labels(test_examples)
		predicted_labels = knn_classifier.predict(feature_vectors)
		num_correct = 0
		num_examples = len(predicted_labels)
		for idx, predicted_label in enumerate(predicted_labels):
			example_id = example_ids[idx]
			expected_label = click_labels[idx]
			if expected_label == predicted_label:
				num_correct += 1
			print "EXPECTED:", expected_label, "PREDICTED:", predicted_label
		accuracy = num_correct / float(num_examples)
		print "ACCURACY:", accuracy
		return accuracy

	def _encoder(self, s):
		if s.isdigit():
			return s
		else:
			lst = []
			for c in s:
				if c.isdigit():
					lst.append(c)
				else:
					lst.append(str(ord(c)-97))
			return ''.join(lst)

def main():
	classifier = Classifier()
	train_examples = classifier.parse_examples("train_100000_lines")
	test_examples = classifier.parse_examples("test_100000_lines")
	knn = classifier.get_knn_classifier(train_examples)
	accuracy = classifier.classify_with_knn(knn, test_examples)
	print "========DONE=========="

if __name__ == "__main__":
    main()







		
