package main;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import algorithms.ItemPopularity;
import algorithms.bpr_plusview;
import algorithms.bpr_plusview_loss;
import algorithms.bpr_plusview_loss_beta;
import data_structure.DenseMatrix;
import data_structure.Rating;
import data_structure.SparseMatrix;
import utils.Printer;

public class BPR_plusview_loss extends main {
	public static void main(String[] argv)throws IOException{
		String method = "mfbpr_view_adv";
		double w0 = 2000;    // learning rate
		boolean showProgress = true;  // whether evaluate after every iter
		boolean showLoss = false;   // whether show loss after every iter						
		double alpha = 0.4;   // Popularity parameter	
		int factors = 10; 	// number of latent factors.
		int maxIter = 100; 	// maximum iterations.
		boolean adaptive = false; 	// Whether to use adaptive learning rate 
		double reg = 0.01; 	// regularization parameters
		int showbound = 400;   // no evaluate outcome before showbound
		int showtime = 1;   // outcome at every showtime iter 
		int paraK = 1;   // DNS paramete
		String viewfile = "data/Tmall_view";
		String datafile = "data/Tmall_purchase";
		double omega = 0;
		
		if (argv.length > 0) {
			w0 = Double.parseDouble(argv[0]);
			showProgress = Boolean.parseBoolean(argv[1]);
			showLoss = Boolean.parseBoolean(argv[2]);
			factors = Integer.parseInt(argv[3]);
			maxIter = Integer.parseInt(argv[4]);
			reg = Double.parseDouble(argv[5]);
			alpha = Double.parseDouble(argv[6]);
			datafile = argv[7];
			viewfile = argv[8];			
			showbound = Integer.parseInt(argv[9]);
			showtime = Integer.parseInt(argv[10]);			
			omega = Double.parseDouble(argv[11]);
		}
		ReadRatings_HoldOneOut(datafile);
		
		SparseMatrix viewmatrix;
		//view file to matrix 
		{
			long startTime = System.currentTimeMillis();
			ArrayList<ArrayList<Rating>> user_ratings = new ArrayList<ArrayList<Rating>>();
			BufferedReader reader = new BufferedReader(
					new InputStreamReader(new FileInputStream(viewfile)));
			String line;
			while((line = reader.readLine()) != null) {
				Rating rating = new Rating(line);
				while (user_ratings.size()  < userCount) { // create a new user
					user_ratings.add(new ArrayList<Rating>());
				}
				user_ratings.get(rating.userId).add(rating);
			}
			reader.close(); 
			System.out.printf("Generate view/buy matrics.");
						startTime = System.currentTimeMillis();
			viewmatrix = new SparseMatrix(userCount, itemCount);
				for (int u = 0; u < userCount; u ++) {
				{
				ArrayList<Rating> ratings = user_ratings.get(u);
				for (int i = ratings.size() - 1; i >= 0; i --) {
					int userId = ratings.get(i).userId;
					int itemId = ratings.get(i).itemId;										
					viewmatrix.setValue(userId, itemId, 1);		
				}
				}
			}			
			System.out.printf("[%s]\n", Printer.printTime(
					System.currentTimeMillis() - startTime));			
			System.out.println ("Data\t" + viewfile);
			System.out.printf("#Ratings\t %d (train)\n", 
					viewmatrix.itemCount());
		}
		
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		evaluate_model(popularity, "Popularity");
		double init_mean = 0;
		double init_stdev = 0.01;

		System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%.6f, w0=%.6f, alpha=%.2f, parak=%d\n",
					method, showProgress, factors, maxIter, reg, w0, alpha, paraK);	
		bpr_plusview_loss bpr = new bpr_plusview_loss(trainMatrix, testRatings, topK, threadNum, 
					factors, maxIter, w0, adaptive, reg, init_mean, init_stdev, showProgress,
					showbound,showtime,viewmatrix,omega);
		evaluate_model(bpr, "bpr_plusview_loss");	
	    	
	}
}