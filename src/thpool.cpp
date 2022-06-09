
#define _POSIX_C_SOURCE 200809L
//#define _GNU_SOURCE
#include <unistd.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#include <cuda_runtime.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include "thpool.h"
#include <sched.h>

#ifdef THPOOL_DEBUG
#define THPOOL_DEBUG 1
#else
#define THPOOL_DEBUG 0
#endif

#if !defined(DISABLE_PRINT) || defined(THPOOL_DEBUG)
#define err(str) fprintf(stderr, str)
#else
#define err(str)
#endif


static volatile int threads_keepalive;
static volatile int threads_on_hold;

/* ========================== PROTOTYPES ============================ */

static int  rt_thread_init(thpool_* thpool_p, struct thread** thread_p, int id);
static int  thread_init(thpool_* thpool_p, struct thread** thread_p, int id);
static void* thread_do(struct thread* thread_p);
static void  thread_hold(int sig_id);
static void  thread_destroy(struct thread* thread_p);

static int   jobqueue_init(jobqueue* jobqueue_p);
static void  jobqueue_clear(jobqueue* jobqueue_p);
static void  jobqueue_push(jobqueue* jobqueue_p, struct job* newjob_p);
static struct job* jobqueue_pull(jobqueue* jobqueue_p);
static void  jobqueue_destroy(jobqueue* jobqueue_p);

static void  bsem_init(struct bsem *bsem_p, int value);
static void  bsem_reset(struct bsem *bsem_p);
static void  bsem_post(struct bsem *bsem_p);
static void  bsem_post_all(struct bsem *bsem_p);
static void  bsem_wait(struct bsem *bsem_p);





/* ========================== THREADPOOL ============================ */

/* Initialise thread pool */
struct thpool_* thpool_init(int num_threads,int idx){

	threads_on_hold   = 0;
	threads_keepalive = 1;

	if (num_threads < 0){
		num_threads = 0;
	}

	/* Make new thread pool */
	thpool_* thpool_p;
	thpool_p = (struct thpool_*)malloc(sizeof(struct thpool_));
	if (thpool_p == NULL){
		err("thpool_init(): Could not allocate memory for thread pool\n");
		return NULL;
	}
	thpool_p->num_threads_alive   = 0;
	thpool_p->num_threads_working = 0;

	/* Initialise the job queue */
	if (jobqueue_init(&thpool_p->jobqueue) == -1){
		err("thpool_init(): Could not allocate memory for job queue\n");
		free(thpool_p);
		return NULL;
	}
	std::cout<<"num_thread : "<<num_threads<<std::endl;
	/* Make threads in pool */
	thpool_p->threads = (struct thread**)malloc(num_threads * sizeof(struct thread *));
	if (thpool_p->threads == NULL){
		err("thpool_init(): Could not allocate memory for threads\n");
		jobqueue_destroy(&thpool_p->jobqueue);
		free(thpool_p);
		return NULL;
	}

	pthread_mutex_init(&(thpool_p->thcount_lock), NULL);
	pthread_cond_init(&thpool_p->threads_all_idle, NULL);

	/* Thread init */
	int n;	//0-2
	int cpu_n = 27;
	
	for (n=0; n<num_threads; n++){

		// #if RT_THREAD
		// 	rt_thread_init(thpool_p, &thpool_p->threads[n], n);
		// #else
		thread_init(thpool_p, &thpool_p->threads[n], n);
		// #endif

		if(n == (num_threads-1)){
			thpool_p->threads[n]->flag = 1;
		}
		else{
			thpool_p->threads[n]->flag = 0;
		}
			
		#if CPU_PINNING
			/* kmsjames 2020 0215 bug fix for pinning each thread on a specified CPU */
			CPU_ZERO(&cpuset);
			if((num_threads*idx+n)>27){
				CPU_SET((cpu_n-(~(num_threads*idx+n))), &cpuset); //only this thread has the affinity for the 'n'-th CPU	
			}else{
				CPU_SET((cpu_n-(num_threads*idx+n)), &cpuset); //only this thread has the affinity for the 'n'-th CPU	
			}
			pthread_setaffinity_np(thpool_p->threads[n]->pthread, sizeof(cpu_set_t), &cpuset);
		#endif

	#if THPOOL_DEBUG
		printf("THPOOL_DEBUG: Created thread %d in pool \n", n);
	#endif
	}

	/* Wait for threads to initialize */
	while (thpool_p->num_threads_alive != num_threads) {}

	return thpool_p;
}


/* Add work to the thread pool */
int thpool_add_work(thpool_* thpool_p, void (*function_p)(void*), void *arg_p){
	job* newjob;
	
	// #if Q_OVERHEAD
	// 	cudaEventCreate(&dq_end);
	// 	cudaEventCreate(&dq_start);
	// #endif
	newjob=(struct job*)malloc(sizeof(struct job));
	if (newjob==NULL){
		err("thpool_add_work(): Could not allocate memory for new job\n");
		return -1;
	}

	/* add function and argument */
	newjob->function=function_p;
	newjob->arg=arg_p;
	// std::cout<<"GPU : "<<(((th_arg*)newjob->arg)->arg)->g_index<<" net index : "<<(((th_arg*)newjob->arg)->arg)->index_n<<" job num : "<<thpool_p->jobqueue.len<<" last : "<<(((th_arg*)newjob->arg)->arg)->cur_round_last<<std::endl;

	/*calculate API num in Q*/
	cal_kernels_enqueue((((th_arg*)newjob->arg)->arg));
	// std::cout<<"q meam : "<<(((th_arg*)newjob->arg)->arg)->layers[(((th_arg*)newjob->arg)->arg)->index].all_api<<std::endl;
	jobqueue_push(&thpool_p->jobqueue, newjob);
	return 0;
}


/* Wait until all jobs have finished */
void thpool_wait(thpool_* thpool_p){
	pthread_mutex_lock(&thpool_p->thcount_lock);
	while (thpool_p->jobqueue.len || thpool_p->num_threads_working) {
		pthread_cond_wait(&thpool_p->threads_all_idle, &thpool_p->thcount_lock);
	}
	pthread_mutex_unlock(&thpool_p->thcount_lock);
}


/* Destroy the threadpool */
void thpool_destroy(thpool_* thpool_p){
	/* No need to destory if it's NULL */
	if (thpool_p == NULL) return ;

	volatile int threads_total = thpool_p->num_threads_alive;

	/* End each thread 's infinite loop */
	threads_keepalive = 0;

	/* Give one second to kill idle threads*/
	double TIMEOUT = 1.0;
	time_t start, end;
	double tpassed = 0.0;
	time (&start);
	while (tpassed < TIMEOUT && thpool_p->num_threads_alive){
		bsem_post_all(thpool_p->jobqueue.has_jobs);
		time (&end);
		tpassed = difftime(end,start);
	}

	/* Poll remaining threads */
	while (thpool_p->num_threads_alive){
		bsem_post_all(thpool_p->jobqueue.has_jobs);
		sleep(1);
	}

	/* Job queue cleanup */
	jobqueue_destroy(&thpool_p->jobqueue);
	/* Deallocs */
	int n;
	for (n=0; n < threads_total; n++){
		thread_destroy(thpool_p->threads[n]);
	}
	free(thpool_p->threads);
	free(thpool_p);
}


/* Pause all threads in threadpool */
void thpool_pause(thpool_* thpool_p) {
	int n;
	for (n=0; n < thpool_p->num_threads_alive; n++){
		pthread_kill(thpool_p->threads[n]->pthread, SIGUSR1);
	}
}


/* Resume all threads in threadpool */
void thpool_resume(thpool_* thpool_p) {
    // resuming a single threadpool hasn't been
    // implemented yet, meanwhile this supresses
    // the warnings
    (void)thpool_p;

	threads_on_hold = 0;
}


int thpool_num_threads_working(thpool_* thpool_p){
	return thpool_p->num_threads_working;
}





/* ============================ THREAD ============================== */


/* Initialize a thread in the thread pool
 *
 * @param thread        address to the pointer of the thread to be created
 * @param id            id to be given to the thread
 * @return 0 on success, -1 otherwise.
 */
static int thread_init (thpool_* thpool_p, struct thread** thread_p, int id){

	*thread_p = (struct thread*)malloc(sizeof(struct thread));
	if (*thread_p == NULL){
		err("thread_init(): Could not allocate memory for thread\n");
		return -1;
	}

	(*thread_p)->thpool_p = thpool_p;
	(*thread_p)->id       = id;

	pthread_create(&(*thread_p)->pthread, NULL, (void *(*)(void*))thread_do, (*thread_p));
	pthread_detach((*thread_p)->pthread);
	return 0;
}


/* Sets the calling thread on hold */
static void thread_hold(int sig_id) {
    (void)sig_id;
	threads_on_hold = 1;
	while (threads_on_hold){
		sleep(1);
	}
}


/* What each thread is doing
*
* In principle this is an endless loop. The only time this loop gets interuppted is once
* thpool_destroy() is invoked or the program exits.
*
* @param  thread        thread that will run this function
* @return nothing
*/


static void* thread_do(struct thread* thread_p){
    

	/* Set thread name for profiling and debuging */
	char thread_name[128] = {0};
	sprintf(thread_name, "thread-pool-%d", thread_p->id);

#if defined(__linux__)
	/* Use prctl instead to prevent using _GNU_SOURCE flag and implicit declaration */
	prctl(PR_SET_NAME, thread_name);
#elif defined(__APPLE__) && defined(__MACH__)
	pthread_setname_np(thread_name);
#else
	err("thread_do(): pthread_setname_np is not supported on this system");
#endif

	/* Assure all threads have been created before starting serving */
	thpool_* thpool_p = thread_p->thpool_p;

	/* Register signal handler */
	struct sigaction act;
	sigemptyset(&act.sa_mask);
	act.sa_flags = 0;
	act.sa_handler = thread_hold;
	if (sigaction(SIGUSR1, &act, NULL) == -1) {
		err("thread_do(): cannot handle SIGUSR1");
	}

	/* Mark thread as alive (initialized) */
	pthread_mutex_lock(&thpool_p->thcount_lock);
	thpool_p->num_threads_alive += 1;
	pthread_mutex_unlock(&thpool_p->thcount_lock);

	// cudaEvent_t dq_start,dq_end;
	// float dq_time;
	// cudaEventCreate(&dq_end);
	// cudaEventCreate(&dq_start);

	while(threads_keepalive){
		bsem_wait(thpool_p->jobqueue.has_jobs);

		if (threads_keepalive){

			pthread_mutex_lock(&thpool_p->thcount_lock);
			thpool_p->num_threads_working++;
			pthread_mutex_unlock(&thpool_p->thcount_lock);

			/* Read job from queue and execute it */
				void (*func_buff)(void*);
				void*  arg_buff;
				
				//check_load==true 면 아직 load 값 확인 전이라는 의미 -> load 값 확인끝날때까지 Q 에서 job을 dequeue 하지않음
				while(check_load){}

				// cudaEventRecord(dq_start);
				job* job_p = jobqueue_pull(&thpool_p->jobqueue);
				
				if (job_p){

					// cal_kernels_dequeue((((th_arg*)job_p->arg)->arg),(((th_arg*)job_p->arg)->arg)->index);
					
					/*DEQ time*/
					// cudaEventRecord(dq_end);
					// cudaEventSynchronize(dq_end);
					// cudaEventElapsedTime(&dq_time, dq_start, dq_end);
					// if((((th_arg*)job_p->arg)->arg)->warming==true){
            		// 	fprintf(((((th_arg*)job_p->arg)->arg)->fp),"%lf\n",dq_time);
					// }

					#if Q_OVERHEAD

						/* CUDA EVENT RECORD -> torch run time error*/
						// if((((th_arg*)job_p->arg)->arg)->warming == true){
						// 	cudaEventRecord((((th_arg*)job_p->arg)->arg)->layers[(((th_arg*)job_p->arg)->arg)->index].q_end);
						// 	cudaEventSynchronize((((th_arg*)job_p->arg)->arg)->layers[(((th_arg*)job_p->arg)->arg)->index].q_end);
						// }

						/* CLOCK_GETTIME */
						long q_time;
						if((((th_arg*)job_p->arg)->arg)->warming == true){
							if (clock_gettime(CLOCK_MONOTONIC, &(((th_arg*)job_p->arg)->arg)->layers[(((th_arg*)job_p->arg)->arg)->index].q_end) == -1) {
        					    perror("clock_gettime");
            					exit(EXIT_FAILURE);
        					}
							q_time = (NANO*((((th_arg*)job_p->arg)->arg)->layers[(((th_arg*)job_p->arg)->arg)->index].q_end).tv_sec - NANO*((((th_arg*)job_p->arg)->arg)->layers[(((th_arg*)job_p->arg)->arg)->index].q_start).tv_sec) + (((((th_arg*)job_p->arg)->arg)->layers[(((th_arg*)job_p->arg)->arg)->index].q_end).tv_nsec - ((((th_arg*)job_p->arg)->arg)->layers[(((th_arg*)job_p->arg)->arg)->index].q_start).tv_nsec);
							//cudaEventRecord(dq_end);
							// cudaEventSynchronize(dq_end);
							// cudaEventElapsedTime(&dq_time, dq_start, dq_end);
							(((th_arg*)job_p->arg)->arg)->layers[(((th_arg*)job_p->arg)->arg)->index].q_time = (double)q_time/MILLI; //ms
							//printf("Q TIME : %d %d %lf\n",(((th_arg*)job_p->arg)->arg)->index_n,(((th_arg*)job_p->arg)->arg)->net->index,(((th_arg*)job_p->arg)->arg)->net->layers[(((th_arg*)job_p->arg)->arg)->net->index].q_time);

						}
					#endif
					
					func_buff = job_p->function;
					arg_buff  = job_p->arg;	//th
					
					func_buff(arg_buff);
					free(job_p);

					pthread_mutex_lock(&mutex_g[(((th_arg*)job_p->arg)->arg)->device->g_index]);
				    gpu_list[(((th_arg*)job_p->arg)->arg)->device->g_index].all_api -= (((th_arg*)job_p->arg)->arg)->all_api;
					gpu_list[(((th_arg*)job_p->arg)->arg)->device->g_index].worker_load -= (((th_arg*)job_p->arg)->arg)->weight;
					pthread_mutex_unlock(&mutex_g[(((th_arg*)job_p->arg)->arg)->device->g_index]);

				}

			pthread_mutex_lock(&thpool_p->thcount_lock);
			thpool_p->num_threads_working--;
			if (!thpool_p->num_threads_working) {
				pthread_cond_signal(&thpool_p->threads_all_idle);
			}
			pthread_mutex_unlock(&thpool_p->thcount_lock);

		}
	}
	pthread_mutex_lock(&thpool_p->thcount_lock);
	thpool_p->num_threads_alive --;
	pthread_mutex_unlock(&thpool_p->thcount_lock);

	return NULL;
}


/* Frees a thread  */
static void thread_destroy (thread* thread_p){
	free(thread_p);
}





/* ============================ JOB QUEUE =========================== */


/* Initialize queue */
static int jobqueue_init(jobqueue* jobqueue_p){
	jobqueue_p->len = 0;
	jobqueue_p->front = NULL;
	jobqueue_p->rear  = NULL;

	jobqueue_p->has_jobs = (struct bsem*)malloc(sizeof(struct bsem));
	if (jobqueue_p->has_jobs == NULL){
		return -1;
	}

	pthread_mutex_init(&(jobqueue_p->rwmutex), NULL);
	bsem_init(jobqueue_p->has_jobs, 0);

	return 0;
}


/* Clear the queue */
static void jobqueue_clear(jobqueue* jobqueue_p){

	while(jobqueue_p->len){
		free(jobqueue_pull(jobqueue_p));
	}

	jobqueue_p->front = NULL;
	jobqueue_p->rear  = NULL;
	bsem_reset(jobqueue_p->has_jobs);
	jobqueue_p->len = 0;

}


/* Add (allocated) job to queue
 */
static void jobqueue_push(jobqueue* jobqueue_p, struct job* newjob){

	pthread_mutex_lock(&jobqueue_p->rwmutex);
	// #if RECORD
	// 	cudaEventRecord((((th_arg*)newjob->arg)->arg)->net->layers[(((th_arg*)newjob->arg)->arg)->net->index].q_start);
	// #endif
	#if Q_OVERHEAD
	 	//cudaEventRecord(dq_start);
		 if (clock_gettime(CLOCK_MONOTONIC, &(((th_arg*)newjob->arg)->arg)->layers[(((th_arg*)newjob->arg)->arg)->index].q_start) == -1) {
        		perror("clock_gettime");
            	exit(EXIT_FAILURE);
        }
	#endif

	newjob->prev = NULL;

	switch(jobqueue_p->len){

		case 0:  /* if no jobs in queue */
					jobqueue_p->front = newjob;
					jobqueue_p->rear  = newjob;
					break;

		default: /* if jobs in queue */
					jobqueue_p->rear->prev = newjob;
					jobqueue_p->rear = newjob;

	}
	jobqueue_p->len++;

	bsem_post(jobqueue_p->has_jobs);
	pthread_mutex_unlock(&jobqueue_p->rwmutex);
}


/* Get first job from queue(removes it from queue)
<<<<<<< HEAD
 *
 * Notice: Caller MUST hold a mutex
=======
>>>>>>> da2c0fe45e43ce0937f272c8cd2704bdc0afb490
 */
static struct job* jobqueue_pull(jobqueue* jobqueue_p){

	pthread_mutex_lock(&jobqueue_p->rwmutex);
	job* job_p = jobqueue_p->front;

	switch(jobqueue_p->len){

		case 0:  /* if no jobs in queue */
		  			break;

		case 1:  /* if one job in queue */
					jobqueue_p->front = NULL;
					jobqueue_p->rear  = NULL;
					jobqueue_p->len = 0;
					break;

		default: /* if >1 jobs in queue */
					jobqueue_p->front = job_p->prev;
					jobqueue_p->len--;
					/* more than one job in queue -> post it */
					bsem_post(jobqueue_p->has_jobs);

	}
	pthread_mutex_unlock(&jobqueue_p->rwmutex);
	// std::cout<<"JOBS : "<<jobqueue_p->len<<std::endl;
	return job_p;
}


/* Free all queue resources back to the system */
static void jobqueue_destroy(jobqueue* jobqueue_p){
	jobqueue_clear(jobqueue_p);
	free(jobqueue_p->has_jobs);
}





/* ======================== SYNCHRONISATION ========================= */


/* Init semaphore to 1 or 0 */
static void bsem_init(bsem *bsem_p, int value) {
	if (value < 0 || value > 1) {
		err("bsem_init(): Binary semaphore can take only values 1 or 0");
		exit(1);
	}
	pthread_mutex_init(&(bsem_p->mutex), NULL);
	pthread_cond_init(&(bsem_p->cond), NULL);
	bsem_p->v = value;
}


/* Reset semaphore to 0 */
static void bsem_reset(bsem *bsem_p) {
	bsem_init(bsem_p, 0);
}


/* Post to at least one thread */
static void bsem_post(bsem *bsem_p) {
	pthread_mutex_lock(&bsem_p->mutex);
	bsem_p->v = 1;
	pthread_cond_signal(&bsem_p->cond);
	pthread_mutex_unlock(&bsem_p->mutex);
}


/* Post to all threads */
static void bsem_post_all(bsem *bsem_p) {
	pthread_mutex_lock(&bsem_p->mutex);
	bsem_p->v = 1;
	pthread_cond_broadcast(&bsem_p->cond);
	pthread_mutex_unlock(&bsem_p->mutex);
}


/* Wait on semaphore until semaphore has value 0 */
static void bsem_wait(bsem* bsem_p) {
	pthread_mutex_lock(&bsem_p->mutex);
	while (bsem_p->v != 1) {
		pthread_cond_wait(&bsem_p->cond, &bsem_p->mutex);
	}
	bsem_p->v = 0;
	pthread_mutex_unlock(&bsem_p->mutex);
}


/*2021-11-03 RT thread hojin*/
static int rt_thread_init (thpool_* thpool_p, struct thread** thread_p, int id){

	pthread_attr_t attr;
	sched_param param;
	int policy = SCHED_RR;

	*thread_p = (struct thread*)malloc(sizeof(struct thread));
	if (*thread_p == NULL){
		err("thread_init(): Could not allocate memory for thread\n");
		return -1;
	}

	(*thread_p)->thpool_p = thpool_p;
	(*thread_p)->id       = id;

	pthread_attr_init(&attr);
	pthread_attr_getschedparam(&attr,&param);
	param.sched_priority = 99;
	pthread_attr_setschedpolicy(&attr,policy);
	pthread_attr_setschedparam(&attr,&param);
	pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);


	pthread_create(&(*thread_p)->pthread, &attr, (void *(*)(void*))thread_do, (*thread_p));
	pthread_detach((*thread_p)->pthread);
	return 0;
}
