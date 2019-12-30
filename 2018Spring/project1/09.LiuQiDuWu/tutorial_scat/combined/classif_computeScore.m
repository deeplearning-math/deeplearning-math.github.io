function out = classif_computeScore(predictedLabels,trueLabels)
nbclass=max(trueLabels(:));

% labels are supposed to be a vector (Nx1)
isCorrectlyClassified = (predictedLabels==trueLabels);
out.confusionMatrix=confusionmatrix(predictedLabels,trueLabels);
for i=1:nbclass
    classMask=(trueLabels==i);
    out.score_per_cat(i)=sum(isCorrectlyClassified(classMask))/sum(classMask);
end
out.score_avg=sum(isCorrectlyClassified)/numel(isCorrectlyClassified);

end