function [predicted_label,nearest_neighbors] = nnpredict(testing_label_vector,...
    testing_instance_matrix,training_label_vector,training_instance_matrix)

 %%predicted_label = svmpredict(testing_label_vector, testing_instance_matrix, model );
 
 d=pairwise_distance(testing_instance_matrix,training_instance_matrix);
 [~,nearest_neighbors]=min(d,[],2);
 predicted_label=training_label_vector(nearest_neighbors);
 