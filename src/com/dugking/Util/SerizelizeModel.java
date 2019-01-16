package com.dugking.Util;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class SerizelizeModel {

	/**
	 * @param model  序列化对象
	 * @param path 序列化路径
	 */
	public static <T> void serlizeModel(T model, String path) {
		
		try {
			FileOutputStream fileOut = new FileOutputStream(path);
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(model);
			out.close();
			fileOut.close();
			System.out.printf("Serialized data is saved in "+path);
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	/**
	 * @param path 反序列化路径
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public static <T> T deSerlizeModel(String path) 
	{
		FileInputStream fileIn;
		  T e = null;
		try {
			fileIn = new FileInputStream(path);
			 ObjectInputStream in = new ObjectInputStream(fileIn);
			    e = (T) in.readObject();
			    in.close();
			    fileIn.close();
			    return e;
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		catch(ClassNotFoundException el){
			el.printStackTrace();
		}
		catch(IOException el)
		{
			el.printStackTrace();
		}
		return e;
	   
	    
	}
	
}
