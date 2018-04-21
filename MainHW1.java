package HomeWork1;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class MainHW1 {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
		
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static Instances filterInstances(Instances data ,int attributeOne, int attributeTwo, int attributeThree) throws Exception {
		Remove remove;
		remove = new Remove();
		String removeIndices = "" + attributeOne + "," + attributeTwo + "," +
				attributeThree + "," + data.numAttributes();
		remove.setAttributeIndices(removeIndices);
		remove.setInvertSelection(true);
		remove.setInputFormat(data);
		return Filter.useFilter(data, remove);
	}


	public static void main(String[] args) throws Exception {
		//load data - CREATE WITH GLOBAL PATH
		Instances trainingData = loadData("wind_training.txt");
		Instances testingData = loadData("wind_testing.txt");

		// Find the best alpha and build the classifier with all attributes
		LinearRegression estimateWindSpeedInMAL = new LinearRegression();
		estimateWindSpeedInMAL.buildClassifier(trainingData);

		System.out.println("The chosen alpha is: " + estimateWindSpeedInMAL.getAlpha());
		System.out.println("Training error with all features is: "
				+ estimateWindSpeedInMAL.calculateMSE(trainingData));
		System.out.println("Test error with all features is: "
				+ estimateWindSpeedInMAL.calculateMSE(testingData));

		// Build the classifier with the same alpha, but with only 3 attributes at a time
		Instances dataWith3Attributes;
		int bestAttributeOne = 0;
		int bestAttributeTwo = 0;
		int bestAttributeThree = 0;
		double bestError, currentError;
		bestError = Double.MAX_VALUE;
		
		System.out.println("\nList of all combination of 3 features and their training error:");
		for (int i = 1; i < testingData.numAttributes(); i++) {
			for (int j = i + 1; j < testingData.numAttributes(); j++) {
				for (int h = j + 1; h < testingData.numAttributes(); h++) {
					dataWith3Attributes = filterInstances(testingData, i, j, h);
					estimateWindSpeedInMAL.buildClassifier(dataWith3Attributes);
					currentError = estimateWindSpeedInMAL.calculateMSE(dataWith3Attributes);
					if (currentError < bestError) {
						bestError = currentError;
						bestAttributeOne = i - 1;
						bestAttributeTwo = j - 1;
						bestAttributeThree = h - 1;
					}
					System.out.println("\t\t" +
										dataWith3Attributes.attribute(0).name() + ", " +
										dataWith3Attributes.attribute(1).name() + ", " +
										dataWith3Attributes.attribute(2).name() + ": \t" +
										currentError);
				}
			}
		}

		System.out.println("\nThe best 3 features are: " +
				trainingData.attribute(bestAttributeOne).name() + ", " +
				trainingData.attribute(bestAttributeTwo).name() + ", " +
				trainingData.attribute(bestAttributeThree).name());
		System.out.println("Training error with best 3 features: " + bestError);
		System.out.print("Test error with best 3 features: ");

		// Rebuild the classifier with the best 3 attributes, calculate MSE with test data
		dataWith3Attributes = filterInstances(trainingData, bestAttributeOne, bestAttributeTwo, bestAttributeThree);
		estimateWindSpeedInMAL.buildClassifier(dataWith3Attributes);

		dataWith3Attributes = filterInstances(testingData, bestAttributeOne, bestAttributeTwo, bestAttributeThree);
		System.out.print(estimateWindSpeedInMAL.calculateMSE(dataWith3Attributes));
		
	}


}
