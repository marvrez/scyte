#include "list.h"

#include <stdlib.h>
#include <string.h>
#include "list.h"

list* make_list()
{
    list* l = malloc(sizeof(list));
    l->size = 0;
    l->head = NULL;
    l->tail = NULL;
    return l;
}

void* list_pop(list* l) 
{
    if(!l->tail) return NULL;
    list_node* n = l->tail;
    void* data = n->data;
    l->tail = l->tail->prev;
    if(l->tail) l->tail->next = NULL;
    free(n);
    --l->size;
    
    return data;
}

void list_prepend(list* l, void* data)
{
    list_node* new = malloc(sizeof(list_node));
    new->data = data, new->next = new->prev = NULL;
    if(l->head) l->head->prev = new;
    new->next = l->head;
    l->head = new;
    if(!l->tail) l->tail = l->head;
    ++l->size;
}

void list_append(list* l, void* data)
{
    list_node* new = malloc(sizeof(list_node));
    new->data = data, new->next = NULL;

    if(!l->tail) {
        l->head = new;
        new->prev = NULL;
    }
    else {
        l->tail->next = new;
        new->prev = l->tail;
    }
    l->tail = new;
    ++l->size;
}

void free_node(list_node* n)
{
    list_node* next;
    while(n) {
        next = n->next;
        free(n);
        n = next;
    }
}

void free_list(list* l)
{
    free_node(l->head);
    free(l);
}

void free_list_contents(list* l)
{
    list_node* n = l->head;
    while(n) {
        free(n->data);
        n = n->next;
    }
}

void** list_to_array(list* l)
{
    void** a = calloc(l->size, sizeof(void*));
    int count = 0;
    list_node* n = l->head;
    while(n) {
        a[count++] = n->data;
        n = n->next;
    }
    return a;
}

void** list_to_reverse_array(list* l)
{
    void** a = calloc(l->size, sizeof(void*));
    int count = 0;
    list_node* n = l->tail;
    while(n) {
        a[count++] = n->data;
        n = n->prev;
    }
    return a;
}
