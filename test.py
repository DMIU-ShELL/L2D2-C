    # Multi-threaded handling of mask send recv
    def send_mask(self, masks_list):
        for item in masks_list:
            mask = item['mask']
            dst_agent_id = item['dst_agent_id']


            buff = torch.ones_like(self.buff_send_mask[dst_agent_id]) * torch.inf

            buff[ParallelComm.META_INF_IDX_PROC_ID] = self.agent_id
            buff[ParallelComm.META_INF_IDX_MSG_TYPE] = ParallelComm.MSG_TYPE_SEND_RESP

            if mask is None:
                buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_NULL
                buff[ParallelComm.META_INF_SZ : ] = torch.inf

            else:
                buff[ParallelComm.META_INF_IDX_MSG_DATA] = ParallelComm.MSG_DATA_MSK
                buff[ParallelComm.META_INF_SZ : ] = mask # NOTE deepcopy?

            print('Buffer to be sent with mask: ', buff, flush=True)
            # Send the mask to the destination agent id
            req = dist.isend(tensor=buff, dst=dst_agent_id)
            req.wait()
            print('Sending mask to agents part 2. All complete!')
        return
    def receive_mask(self, best_agent_id):
        received_mask = None
        print(Fore.GREEN + 'Best Agent: ', best_agent_id)
        if best_agent_id:
            # We want to get the mask from the best agent
            buff = torch.ones_like(self.buff_recv_mask[0]) * torch.inf
            print(buff, len(buff))
            # Receive the buffer containing the mask. Wait for 10 seconds to make sure mask is received
            print('Mask recv start')
            req = dist.irecv(tensor=buff, src=best_agent_id)
            req.wait()
            print('Mask recv end')
            #time.sleep(ParallelComm.SLEEP_DURATION)

            # If the buffer was a null response (which it shouldn't be)
            # meep
            if self._null_message(buff):
                # Shouldn't reach this hopefully :^)
                return None

            # otherwise return the mask
            elif buff[ParallelComm.META_INF_IDX_MSG_DATA] == ParallelComm.MSG_DATA_MSK:
                if buff[ParallelComm.META_INF_IDX_PROC_ID] == best_agent_id:
                    return buff[ParallelComm.META_INF_IDX_MASK_SZ : ]


            #received_mask = self.receive_mask_response(best_agent_id)
            print(Fore.GREEN + 'Received mask length: ', len(received_mask))

            # Reset the best agent id for the next request
            best_agent_id = None

        return received_mask, best_agent_id

    def send_recv_mask(self, masks_list, best_agent_id):
        #print('Send Recv Mask Function', len(mask), dst_agent_id, best_agent_id)
        pool1 = mp.pool.ThreadPool(processes=1)
        pool2 = mp.pool.ThreadPool(processes=1)

        _ = pool1.apply_async(self.send_mask, (masks_list,))
        result = pool2.apply_async(self.receive_mask, (best_agent_id,))

        return result.get()
