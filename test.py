import matplotlib.pyplot as plt

N_values = [1, 2, 4, 8, 12, 16, 20]
execution_times = [3.2843177318573, 3.7008888721466064, 4.289881467819214, 6.688939094543457, 9.958037614822388, 11.427663564682007, 14.93051791191101]
execution_times1 = [2.6695480346679688, 5.203624963760376, 10.701014757156372, 21.853699445724487, 31.40264630317688, 42.56728219985962, 53.669623613357544]

plt.figure()
plt.plot(N_values, execution_times, marker='o', label="8 processes")
plt.plot(N_values, execution_times1, marker='o', label="default")
plt.xlabel("Number of Scenarios (N)")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time vs Number of Scenarios ")
plt.grid(True)
plt.legend()
plt.show()