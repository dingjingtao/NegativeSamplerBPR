package algorithms;

import data_structure.Rating;
import data_structure.SparseMatrix;
import data_structure.DenseVector;
import data_structure.DenseMatrix;
import data_structure.Pair;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import utils.Printer;

/**
 * Implement BPR with view data (BPR+View_loss).
 * Ding, Yu, et all. "Sampler Design for Bayesian Personalized Ranking by Leveraging View Data"
 * 
 * @author dingjingtao
 *
 */
public class bpr_plusview_loss extends TopKRecommender {
	/** Model priors to set. */
	int factors = 10; 	// number of latent factors.
	int maxIter = 100; 	// maximum iterations.
	double lr = 0.01; 		// Learning rate
	boolean adaptive = false; 	// Whether to use adaptive learning rate 
	double reg = 0.01; 	// regularization parameters
	double init_mean = 0;  // Gaussian mean for init V
	double init_stdev = 0.1; // Gaussian std-dev for init V
	int showbound = 0;
	int showtime = 1;
	/** Model parameters to learn */
	public DenseMatrix U;	// latent vectors for users
	public DenseMatrix V;	// latent vectors for items
	int usercount =0;
	int itemcount =0;
	public Integer [][]viewdata;
	public Integer [][]buydata;
	boolean showProgress;
	public String onlineMode = "u";
	SparseMatrix viewMatrix;
	double omega = 0;
	Random rand = new Random();
	
  
	public bpr_plusview_loss(SparseMatrix trainMatrix, ArrayList<Rating> testRatings,
			int topK, int threadNum, int factors, int maxIter, double lr, boolean adaptive, double reg, 
			double init_mean, double init_stdev, boolean showProgress,
			int showbound ,int showtime ,SparseMatrix viewMatrix,double omega) {
		super(trainMatrix, testRatings, topK, threadNum);
		this.factors = factors;
		this.maxIter = maxIter;
		this.lr = lr;
		this.adaptive = adaptive;
		this.reg = reg;
		this.init_mean = init_mean;
		this.init_stdev = init_stdev;
		this.showProgress = showProgress;
		this.showbound = showbound;
		this.showtime = showtime;
		this.viewMatrix = viewMatrix;
		this.omega = omega;
		// Init model parameters
		U = new DenseMatrix(userCount, factors);
		V = new DenseMatrix(itemCount, factors);
		U.init(init_mean, init_stdev);
		V.init(init_mean, init_stdev);
		this.usercount = userCount;
		this.itemcount = itemCount;
		
	}
	
	
	public void setUV(DenseMatrix U, DenseMatrix V) {
		this.U = U.clone();
		this.V = V.clone();
	}
	
	public void buildModel() {	
		int nonzeros = trainMatrix.itemCount();
		double hr_prev = 0;
		
		buydata = new Integer[userCount][];
		viewdata = new Integer[userCount][];
		for (int i =0;i<userCount;i++) {
			ArrayList<Integer> itemList = trainMatrix.getRowRef(i).indexList();
			ArrayList<Integer> viewList = viewMatrix.getRowRef(i).indexList();
			buydata[i] = itemList.toArray(new Integer [itemList.size()]);
			viewdata[i] = viewList.toArray(new Integer [viewList.size()]);
		}
		System.out.println("array has been available");
		
		for (int iter = 0; iter < maxIter; iter ++) {
			Long start = System.currentTimeMillis();
			rand = new Random();
			for (int s = 0; s < nonzeros; s ++) { 
				// sample a user				
				int u = rand.nextInt(userCount); 
				if (buydata[u].length ==0 && viewdata[u].length ==0){
					continue;
				}else if(viewdata[u].length ==0) {
					int i = buydata[u][rand.nextInt(buydata[u].length)];
					update_ui(u, i);
					continue;
				}else if (buydata[u].length ==0) {
					int i = viewdata[u][rand.nextInt(viewdata[u].length)];
					update_ui(u, i);
					continue;			
				}else {
					int i = buydata[u][rand.nextInt(buydata[u].length)];
					int j = viewdata[u][rand.nextInt(viewdata[u].length)];
					update_uij(u, i, j);
					continue;
				}
			}
			if (showProgress)
				if(iter >= showbound  | iter %20 == 0)
				showProgress(iter, start, testRatings);
			
			// Adjust the learning rate
			if (adaptive) {
				if (!showProgress)	evaluate(testRatings);
				double hr = ndcgs.mean();
				lr = hr > hr_prev ? lr * 1.05 : lr * 0.5;
				hr_prev = hr;
			}			
		} 		
	}
	
	public void runOneIteration() {
		int nonzeros = trainMatrix.itemCount();
		rand = new Random();
		// Each training epoch
		for (int s = 0; s < nonzeros; s ++) { 
			// sample a user
			int u = rand.nextInt(userCount); 
			ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
			if (itemList.size() == 0)	continue;
			// sample a positive item
			int i = itemList.get(rand.nextInt(itemList.size())); 
			
			// One SGD step update
			update_ui(u, i);

		}
	}
	
	//One SGD step for a positive instance.
	private void update_ui(int u, int i) {
		//decide j from uniform or view
		double p = rand.nextDouble();
		int j = rand.nextInt(itemCount);
		
		while (trainMatrix.getValue(u, j) != 0 && viewMatrix.getValue(u, j) != 0 ) {
			j = rand.nextInt(itemCount);
		}
		// BPR update rules
		double y_pos = predict(u, i);  // target value of positive instance
		double y_neg = predict(u, j);  // target value of negative instance
		double mult = -partial_loss(y_pos - y_neg);   
		for (int f = 0; f < factors; f ++) {
			double grad_u = V.get(i, f) - V.get(j, f);
			U.add(u, f, -lr * (mult * grad_u + reg * U.get(u, f)));
   			double grad = U.get(u, f);
			V.add(i, f, -lr * (mult * grad + reg * V.get(i, f)));
			V.add(j, f, -lr * (-mult * grad + reg * V.get(j, f)));
		}
	}
	
	//One SGD step for a positive instance and an intermediate instance.
	private void update_uij(int u, int i, int j) {
		//decide j from uniform or view
		double p = rand.nextDouble();
		int k = rand.nextInt(itemCount);
		while (trainMatrix.getValue(u, k) != 0 && viewMatrix.getValue(u, k) != 0 ) {
			k = rand.nextInt(itemCount);
		}
		
		// BPR update rules
		double y_pos = predict(u, i);  // target value of positive instance
		double y_intem = predict(u, j);  // target value of negative instance
		double y_neg = predict(u, k);  // target value of positive instance
		
		double mult1 = -partial_loss(y_pos - y_intem);  
		double mult2 = -partial_loss(y_intem - y_neg);  
		double omega_adv = omega;
		for (int f = 0; f < factors; f ++) {
			double grad_u1 = V.get(i, f) - V.get(j, f);
			double grad_u2 = V.get(j, f) - V.get(k, f);
			U.add(u, f, -lr * (omega_adv * mult1 * grad_u1 + (1 - omega_adv) * mult2 * grad_u2 + reg * U.get(u, f)));
   			double grad = U.get(u, f);
			V.add(i, f, -lr * (omega_adv * mult1 * grad + reg * V.get(i, f)));
			V.add(j, f, -lr * (- omega_adv * mult1 * grad + (1-omega_adv) * mult2 * grad + reg * V.get(j, f)));
			V.add(k, f, -lr * (- (1 - omega_adv) * mult2 * grad + reg * V.get(k, f)));
		}
	}
		
	@Override
	public double predict(int u, int i) {
		return U.row(u, false).inner(V.row(i, false));
	}
	
  // Partial of the ln sigmoid function used by BPR.
  private double partial_loss(double x) {
    double exp_x = Math.exp(-x);
    return exp_x / (1 + exp_x);
  }

  // Implement the Recsys08 method: Steffen Rendle, Lars Schmidt-Thieme,
  // "Online-Updating Regularized Kernel Matrix Factorization Models"
	public void updateModel(int u, int item) {
		trainMatrix.setValue(u, item, 1);
		rand = new Random();
		
		// user retrain
		ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
		for (int iter = 0; iter < maxIterOnline; iter ++) {
			Collections.shuffle(itemList);
			
			for (int s = 0; s < itemList.size(); s ++) {
				// retrain for the user or for the (user, item) pair
				int i = onlineMode.equalsIgnoreCase("u") ? itemList.get(s) : item;
				// One SGD step update
				update_ui(u, i);
			}
		}
		
	}
}
