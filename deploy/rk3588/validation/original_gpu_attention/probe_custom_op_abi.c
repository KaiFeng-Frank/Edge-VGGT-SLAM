/* Inspect installed SDK structure layout; no model or SDK calls. */
#include <stddef.h>
#include <stdio.h>
#include "rknn_custom_op.h"
#define SIZE(T) printf("\"" #T "\":%zu",sizeof(T))
#define FIELD(T,F) printf(",\"" #T "." #F "\":%zu",offsetof(T,F))
int main(void){
 printf("{");SIZE(rknn_custom_op);FIELD(rknn_custom_op,version);FIELD(rknn_custom_op,target);FIELD(rknn_custom_op,op_type);FIELD(rknn_custom_op,cl_kernel_name);FIELD(rknn_custom_op,cl_kernel_source);FIELD(rknn_custom_op,cl_source_size);FIELD(rknn_custom_op,cl_build_options);FIELD(rknn_custom_op,init);FIELD(rknn_custom_op,prepare);FIELD(rknn_custom_op,compute);FIELD(rknn_custom_op,compute_native);FIELD(rknn_custom_op,destroy);
 printf(",");SIZE(rknn_custom_op_context);FIELD(rknn_custom_op_context,target);FIELD(rknn_custom_op_context,internal_ctx);FIELD(rknn_custom_op_context,gpu_ctx);FIELD(rknn_custom_op_context,priv_data);
 printf(",");SIZE(rknn_custom_op_tensor);FIELD(rknn_custom_op_tensor,attr);FIELD(rknn_custom_op_tensor,mem);
 printf(",");SIZE(rknn_custom_op_attr);FIELD(rknn_custom_op_attr,name);FIELD(rknn_custom_op_attr,dtype);FIELD(rknn_custom_op_attr,n_elems);FIELD(rknn_custom_op_attr,data);
 printf(",");SIZE(rknn_gpu_op_context);FIELD(rknn_gpu_op_context,cl_context);FIELD(rknn_gpu_op_context,cl_command_queue);FIELD(rknn_gpu_op_context,cl_kernel);
 printf(",");SIZE(rknn_tensor_attr);printf(",");SIZE(rknn_tensor_mem);FIELD(rknn_tensor_mem,virt_addr);FIELD(rknn_tensor_mem,fd);FIELD(rknn_tensor_mem,offset);FIELD(rknn_tensor_mem,size);
 printf("}\n");return 0;
}
