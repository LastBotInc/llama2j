package com.lastbot.llama2j;

public class Context {
    final ContextCUDA cuda;
    final ContextCPU cpu;

    final Target target;

    public Context(Target target) {
        this.target = target;
        this.cpu = target.CPU() ? new ContextCPU("contextCPU0", 0, 20) : null;
        this.cuda = target.CUDA() ? new ContextCUDA("contextCUDA0", 0, 20) : null;
    }
}
