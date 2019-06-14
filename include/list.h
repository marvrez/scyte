#ifndef LINKED_LIST_H
#define LINKED_LIST_H

typedef struct list_node {
    void* data;
    struct list_node* next;
    struct list_node* prev;
} list_node;

typedef struct list {
    int size;
    list_node* head;
    list_node* tail;
} list;

list* make_list();
int list_find(list *l, void *val);

void list_append(list*, void*);
void list_prepend(list*, void*);

void* list_pop(list*);

void free_list_contents(list *l);
void** list_to_array(list *l);
void** list_to_reverse_array(list *l);
void free_list(list *l);

#endif
